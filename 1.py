# import the necessary libraries
import io
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
from args import get_parser
import pickle
from model import get_model
from torchvision import transforms
from output_utils import prepare_output
from PIL import Image

data_dir = 'Foodimg2Ing/data/'
ngrs_vocab = pickle.load(open(os.path.join(data_dir, 'ingr_vocab.pkl'), 'rb'))
vocab = pickle.load(open(os.path.join(data_dir, 'instr_vocab.pkl'), 'rb'))
ingr_vocab_size = len(ngrs_vocab)
instrs_vocab_size = len(vocab)
output_dim = instrs_vocab_size

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write('')
        with col2:
            st.image(image_data,width=300)
        with col3:
            st.write('')
                
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def load_model(data_dir):

    args = get_parser()
    args.maxseqlen = 15
    args.ingrs_only=False
    model = get_model(args,ingr_vocab_size, instrs_vocab_size)
    # Load the pre-trained model parameters
    model_path = os.path.join(data_dir, 'modelbest.ckpt')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    model.ingrs_only = False
    model.recipe_only = False
    return model



def predict(model,image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    greedy = [True, False, False, False]
    beam = [-1, -1, -1, -1]
    temperature = 1.0
    numgens = len(greedy)

    num_valid = 1
    for i in range(numgens):
        with torch.no_grad():
            outputs = model.sample(input_batch, greedy=greedy[i], 
                                       temperature=temperature, beam=beam[i], true_ingrs=None)
                
        ingr_ids = outputs['ingr_ids'].cpu().numpy()
        recipe_ids = outputs['recipe_ids'].cpu().numpy()
                
        outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ngrs_vocab, vocab)

        if valid['is_valid'] or show_anyways:
            st.write('**RECIPE**',num_valid)  
            num_valid+=1
            #print ("greedy:", greedy[i], "beam:", beam[i])
            col1, col2, col3 = st.columns([1,1,3])
            with col1:
                st.write('\nTitle:',outs['title'])
            with col2:
                st.write('\nIngredients:',', '.join(outs['ingrs']))
            with col3:
                st.write('\nInstructions:\n','\n-'+'\n\n-'.join(outs['recipe']))
            

    
def main():
    image = Image.open("Food_header.jpg").resize((680, 150))
    st.image(image)
    st.markdown('<div style="text-align: center;"><h1>PIC\'N BASKET</h1><div>',unsafe_allow_html=True)
    st.text("")
    st.markdown('<div style="font-face: Playfair Display;">Add the image to get the recipe !!!</div>',unsafe_allow_html=True)
    st.text("")
    model = load_model(data_dir)
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        predict(model, image)
        
    with st.sidebar:
        #st.expander("ABOUT THE APP", expanded=True)
        st.markdown('<div style="text-align: center;"><h1>ABOUT THE APP</h1><div>',unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify;">This app try to develop an efficient recipe recommendation system that is able to tell the top 5 relevant recipes based on relevancy and nutrient level.<div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify;"> Our system guesses ingredients and recipes at the same time. We test the whole system on different pre trained models like VGG, Resnet and Inception Net.<div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: justify;"><br><b>DATASET USED:</b> Recipe 1M<br><div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center;"><br><b>PROJECT DETAIL:</b><br>This Project is made in partial fulfilment of the requirements for the award of <br>Bachelor of Engineering <br> IN <br> COMPUTER SCIENCE AND ENGINEERING<br><br><div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center;"><b>MADE BY:</b><div>',unsafe_allow_html=True)
        st.markdown('<div style="text-align: center;">Kriti(Co19335)<br>Karan(Co19332)<div>',unsafe_allow_html=True)
        st.markdown('<div style="text-align: center;"><i>Chandigarh College of Engineering & Technology</i><br><br><div>',unsafe_allow_html=True)        
        col1, col2, col3 = st.columns([1,3,1])
        image = Image.open("logo.png").resize((200, 90))
        with col1:
            st.write('')
        with col2:
            st.image(image)
        with col3:
            st.write('')
            
if __name__ == '__main__':
    main()
    
