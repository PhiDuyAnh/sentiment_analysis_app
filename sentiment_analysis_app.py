from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional
from datetime import datetime
import os
import streamlit as st

class AppModel:
    def __init__(self, model_name):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def classify(self, prompt) -> Optional[str]:
        try:
            classifier = pipeline('sentiment-analysis', model=self.model, tokenizer=self.tokenizer)
            results = classifier(prompt)
            return results
        except Exception as e:
            print(f'Error: {str(e)}')
    
def main():
    st.set_page_config(
        page_title='Text Classification'
    )
    st.markdown("<h1 style='text-align: center;'>Type Your Sentence! 🤗</h1>", unsafe_allow_html=True)
    
    # First section
    st.header('English')
    prompt_english = st.text_area(
        label='Enter your sentence in English and I will classify it!',
        placeholder='Type your sentence here...',
        help = 'Your sentence will be analyzed for sentiment classification'
        )
    button1 = st.button('Classify the sentence', type='primary')
    result_placeholder1 = st.empty()

    # Second section
    st.header('Tiếng Việt')
    prompt_vn = st.text_area(
        label='Nhập câu của bạn bằng Tiếng Việt và tôi sẽ phân loại nó!',
        placeholder='Nhập câu tại đây...',
        help='Câu của bạn sẽ được phân tích để phân loại cảm xúc')
    button2 = st.button('Phân loại câu', type='primary')
    result_placeholder2 = st.empty()

    if button1:
        with result_placeholder1.container():
            with st.spinner('Please wait a moment...'):
                model = AppModel('distilbert/distilbert-base-uncased-finetuned-sst-2-english')
                results = model.classify(prompt_english)[0]
                chat_message = st.chat_message('assistant')
                chat_message.markdown(f'The sentence is {results["label"].capitalize()} with a probability of {results["score"]:.2%}!')
                
                os.write(1, f'============================{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}============================\n'.encode())
                os.write(1, f'{prompt_english}\n'.encode())
                os.write(1, f'{results["label"]}, {results["score"]}\n\n'.encode())

    if button2:
        with result_placeholder2.container():
            with st.spinner('Vui lòng đợi trong giây lát...'):
                model = AppModel('wonrax/phobert-base-vietnamese-sentiment')
                results = model.classify(prompt_vn)[0]
                chat_message = st.chat_message('assistant')
                if results['label'] == 'POS':
                    chat_message.markdown(f'Câu đã nhập mang sắc thái Tích cực với xác suất {results["score"]:.2%}!')
                elif results['label'] == 'NEG':
                    chat_message.markdown(f'Câu đã nhập mang sắc thái Tiêu cực với xác suất {results["score"]:.2%}!')
                else:
                    chat_message.markdown(f'Câu đã nhập mang sắc thái Trung lập với xác suất {results["score"]:.2%}!')

                os.write(1, f'============================{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}============================\n'.encode())
                os.write(1, f'{prompt_vn}\n'.encode())
                os.write(1, f'{results["label"]}, {results["score"]}\n\n'.encode())
                

if __name__ == '__main__':
    main()

