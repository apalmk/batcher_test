import tensorflow as tf
import os
import numpy
from numpy import dot
from numpy.linalg import norm
import pickle
import tensorflow.keras.backend as K
from transformers import RobertaTokenizer, TFRobertaModel
from transformers import XLNetTokenizer, TFXLNetModel
from transformers import AutoTokenizer, TFAutoModel
#fast tokenizers
from transformers import DistilBertTokenizerFast, TFDistilBertModel
from transformers import BertTokenizerFast, TFBertModel


def generate_dataset(outputDir,lm_input_style, max_l, hugging_face_model,voc,batch_size,unk_wi,fll):
    '''
    params:
    max_l: (max length of the sentence) #should be same across all the files: take the max sentence length with some upper cap
    '''
    with open(voc,'rb') as f:
        voc=pickle.load(f)

    voc=voc['word_vocabulary']

    files = tf.io.matching_files(os.path.join(outputDir, lm_input_style+'*.csv'))
    shards = tf.data.Dataset.from_tensor_slices(files)
    dataset = shards.interleave(tf.data.TextLineDataset, cycle_length=5,num_parallel_calls=tf.data.experimental.AUTOTUNE)

    defaultData = [str()]*2+[int()]+[int()]*(max_l-3)
    # dataset = tf.data.experimental.CsvDataset(filenames=fnames,
    #                                       record_defaults=[tf.string]*2+[tf.int32]+[tf.int32]*(max_l-3),
    #                                       header=False)

    #Wrap 'encode' function inside 'tf.py_function' so that it could be used with 'map' method.
    #load huggingfae tokenizer

    tokenizer, model = initialize_hugface_model(hugging_face_model)


    def encode(target, text, target_role, input_roles):
        #TODO: Need to clean and organize this

        #Get indices of not (-1) roles (head words)
        minus_one = tf.constant(-1, dtype=tf.int32)
        where = tf.not_equal(input_roles, minus_one)
        indices = tf.where(where) #indices of the head words in the whole sentence
        index_of_headwords = tf.reshape(indices, [-1]).numpy().tolist()
        # map each word to head words and it's subtokens
        # generate output_ids

        input_tokens =tokenizer(tf.compat.as_str(text.numpy()).split(),
                        return_tensors = "tf",
                        is_split_into_words=True,
                        add_special_tokens=False,
                        return_offsets_mapping=True)


        extract_first_token = []
        idx = -1
        idx_headword = -1
        for subtoken in input_tokens.offset_mapping[0]:
            idx += 1
            if subtoken[0] == 0:
                 extract_first_token.append(idx)
                 idx_headword+=1
                 if idx_headword not in index_of_headwords:
                     extract_first_token.remove(idx)

        target_ids = tokenizer.encode(tf.compat.as_str(target.numpy()), add_special_tokens=False)

        #Get the embedding corresponding to each subtoken from the LM
        input_embedding = model(input_tokens)[0]
        target_embedding = model(tf.constant(target_ids)[None, :])[0]

        #If the user choses to take the first token
        if fll==0:
        #return only embeddings for head words
            input_embedding = tf.squeeze(input_embedding, [0])
            input_embedding = tf.gather(input_embedding, extract_first_token)
            input_embedding = tf.expand_dims(input_embedding, axis=0)
            input_embedding = tf.reshape(input_embedding, [1,-1,768])
            target_embedding = tf.squeeze(target_embedding, [0])
            target_embedding = tf.gather(target_embedding, [0])

        elif fll==1: #extracting last token not implemented yet - let's test what we have and see if it speeds up the runs
            input_embedding = tf.squeeze(input_embedding, [0])
            input_embedding = tf.gather(input_embedding, extract_first_token) #replace with extract_last_token later
            input_embedding = tf.expand_dims(input_embedding, axis=0)
            input_embedding = tf.reshape(input_embedding, [1,-1,768])
            target_embedding = tf.squeeze(target_embedding, [0])
            target_embedding = tf.gather(target_embedding, [-1])

        #pad end of input_embedding for return
        ind = 6 - len(extract_first_token)
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

        # target=tf.expand_dims(target,axis=1)
        # text=tf.expand_dims(text,axis=1)
        # target_role=tf.expand_dims(target_role,axis=1)
        #input_roles = tf.stack(args, axis=1) #stack all the other roles into one input roles tensor with dtype int32

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

    #dataset = dataset.batch(batch_size).map(encode_pyfn, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(3)

    dataset = dataset.map(lambda x: tf.io.decode_csv(x, record_defaults=defaultData),num_parallel_calls=tf.data.experimental.AUTOTUNE).map(encode_pyfn, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(3)

    return dataset

def initialize_hugface_model(hugging_face_model):
    # if hugging_face_model == "xlnet":
    #     tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    #     model = TFXLNetModel.from_pretrained('xlnet-base-cased')
    # elif hugging_face_model == "roberta":
    #     tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    #     model = TFRobertaModel.from_pretrained('roberta-base')
    # elif hugging_face_model == "ernie":
    #     tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-en")
    #     model = TFAutoModel.from_pretrained("nghuyong/ernie-2.0-en")

    #FAST TOKENIZERS
    if hugging_face_model == "distilbert":
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
    elif hugging_face_model == "bert":
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        model = TFBertModel.from_pretrained('bert-base-cased')


    else:
        raise ValueError('Invalid embedding type')
    return tokenizer, model


if __name__=='__main__':
    #use this to check, debug, and print the tensor values
    import time

    train_dataset = generate_dataset('','fullsen', 102, "distilbert","description3",10,50001,0)

    for i in train_dataset.take(1):
        print(i)
