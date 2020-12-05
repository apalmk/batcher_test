import tensorflow as tf
import os
from numpy import dot
from numpy.linalg import norm
import tensorflow.keras.backend as K
from transformers import RobertaTokenizer, TFRobertaModel, XLNetTokenizer, TFXLNetModel, AutoTokenizer, TFAutoModel
import pickle

def generate_dataset(fnames, max_l, hugging_face_model,voc,batch_size,unk_wi,fll):
    '''
    params:
    max_l: (max length of the sentence) #should be same across all the files: take the max sentence length with some upper cap
    '''
    with open(voc,'rb') as f:
        voc=pickle.load(f)

    voc=voc['word_vocabulary']
    dataset = tf.data.experimental.CsvDataset(filenames=fnames,
                                          record_defaults=[tf.string]*2+[tf.int32]+[tf.int32]*(max_l-3),
                                          header=False)
    #Wrap 'encode' function inside 'tf.py_function' so that it could be used with 'map' method.
    #load huggingfae tokenizer

    tokenizer, model = initialize_hugface_model(hugging_face_model)


    def encode(target, text, target_role, input_roles):
        #TODO: Need to clean and organize this

        #Get indices of not (-1) roles (head words)
        minus_one = tf.constant(-1, dtype=tf.int32)
        where = tf.not_equal(input_roles, minus_one)
        indices = tf.where(where) #indices of the head words in the whole sentence
        # map each word to head words and it's subtokens
        # generate output_ids

        idx = 0
        enc =[tokenizer.encode(x, add_special_tokens=False) for x in tf.compat.as_str(text.numpy()).split()]

        print("this is the encoded sentence",enc)

        desired_output = []
        for token in enc:
            tokenoutput = []
            for ids in token:
                tokenoutput.append(idx)
                idx +=1
            desired_output.append(tokenoutput)
        first_token = [] #list containing first subtokens of the head words
        last_token = [] #list containing last subtokens of the head words
        pos_len = [] #length of each headword subtokens list
        for pos, idx in enumerate(desired_output):
            if pos in indices:
                first_token.append(idx[0])
                last_token.append(idx[-1])
                pos_len.append(len(idx))
        
        #encode string to tokenized word ids of the LM
        input_ids = tokenizer.encode(tf.compat.as_str(text.numpy()), add_special_tokens=False)

        print("These are input IDs",input_ids)
        target_ids = tokenizer.encode(tf.compat.as_str(target.numpy()), add_special_tokens=False)

        #Get the embedding corresponding to each subtoken from the LM
        input_embedding = model(tf.constant(input_ids)[None, :])[0]
        target_embedding = model(tf.constant(target_ids)[None, :])[0]

        #If the user choses to take the first token
        if fll==0:
        #return only embeddings for head words
            input_embedding = tf.squeeze(input_embedding, [0])
            input_embedding = tf.gather(input_embedding, first_token)
            input_embedding = tf.expand_dims(input_embedding, axis=0)
            input_embedding = tf.reshape(input_embedding, [1,-1,768])
            target_embedding = tf.squeeze(target_embedding, [0])
            target_embedding = tf.gather(target_embedding, [0])
            
        elif fll==1:
            input_embedding = tf.squeeze(input_embedding, [0])
            input_embedding = tf.gather(input_embedding, last_token)
            input_embedding = tf.expand_dims(input_embedding, axis=0)
            input_embedding = tf.reshape(input_embedding, [1,-1,768])
            target_embedding = tf.squeeze(target_embedding, [0])
            target_embedding = tf.gather(target_embedding, [-1])
            
        #pad end of input_embedding for return
        ind = 6 - len(first_token)
        paddings = [[0, 0], [0, ind], [0,0]]
        input_embedding = tf.pad(input_embedding, paddings, 'CONSTANT', constant_values=0.0)
        
        #Tensor to compare and remove the -1 roles
        v = tf.constant(-1)

        #Removing the -1 roles
        roles1 = tf.cast(tf.squeeze(tf.gather(input_roles,tf.where(tf.not_equal(input_roles, v))),[1]), dtype=tf.int32, name=None)

        #Reshaping the target role
        target_role = tf.reshape(target_role,shape=(1,))

        # Adding the target role to the roles to find out the left out roles to append
        r1 = tf.cast(tf.concat([roles1,target_role], axis=0),dtype=tf.int32)

        #Generating False boolean map
        fmask = tf.constant([False]*7, dtype=tf.bool)
        
        #Getting the values from the indices of the roles in the mask
        fl=tf.gather(fmask,indices=r1)

        mask = tf.tensor_scatter_nd_update(tf.constant([True]*7, dtype=tf.bool), tf.expand_dims(r1,axis=1), fl)

        #Generating all the roles in the dataset
        rang=tf.range(7, delta=1, dtype=tf.int32, name='range')

        role_input=tf.concat([roles1,tf.boolean_mask(rang, mask)],axis=0)

        #Getting the target word ID from the dictionary

        #if the word is not in the vocab
        w=target.numpy().decode("utf-8")
        if w not in voc.keys():
            target_word_id = tf.constant(unk_wi)
        else:
            target_word_id=tf.constant(voc[w])

        #Removing unnecessary dimension from inputs
        input_embedding=tf.squeeze(input_embedding,axis=0)
        #target_embedding=tf.squeeze(target_embedding,axis=0)
        target_role_input = target_role

        #Removing the dimension from target role for sending it as an output
        target_role_output = tf.squeeze(target_role)

        # target_word_id=tf.squeeze(tf.reshape(target_word_id, shape=(-1,1), name='n1'),axis=1)
        # target_role_output=tf.squeeze(tf.reshape(target_role_output, shape=(-1,1), name='n2'),axis=1)

        # print("tw ID:", target_word_id)
        # print("tr ID:", target_role)
        #target_word_id.set_shape([1])
        #target_role_output.set_shape([1])

        return input_embedding,target_embedding, target_word_id, role_input, target_role_input, target_role_output

    def encode_pyfn(target, text, target_role, *args):
        input_roles = tf.stack(args, axis=0) #stack all the other roles into one input roles tensor with dtype int32
        ie,te,twi,ri,tr,tro = tf.py_function(encode,
                                          inp=[target, text, target_role, input_roles],
                                          Tout=(tf.float32,tf.float32,tf.int32,tf.int32,tf.int32,tf.int32))

        return (
            {'input_roles': ri,
             'input_words': ie,
             'target_role': tr,
             'target_word': te},
            {'r_out': tro,
             'w_out': twi}
        )

    dataset = dataset.map(encode_pyfn).batch(batch_size)

    return dataset

def initialize_hugface_model(hugging_face_model):
    if hugging_face_model == "xlnet":
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        model = TFXLNetModel.from_pretrained('xlnet-base-cased')
    elif hugging_face_model == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = TFRobertaModel.from_pretrained('roberta-base')
    elif hugging_face_model == "ernie":
        tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-en")
        model = TFAutoModel.from_pretrained("nghuyong/ernie-2.0-en")
    else:
        raise ValueError('Invalid embedding type')
    return tokenizer, model


if __name__=='__main__':
    #use this to check, debug, and print the tensor values
    train_dataset = generate_dataset(['NN_dis_dev1.csv','NN_dis_dev10.csv'], 14, "xlnet","description3",1,50001,0)

    for i in train_dataset.take(1):
        print(i[1]['r_out'].shape,"\n")


