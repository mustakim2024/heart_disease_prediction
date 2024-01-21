# Mebgimpor library
import pandas as pd
import streamlit as st
import pickle
#import urllib.request
from PIL import Image

# Menghilangkan warning
import warnings
warnings.filterwarnings("ignore")

image = Image.open('gambar_jantung.png')
st.image(image, use_column_width=True)

# Membuat judul
st.title('Selamat Datang di Aplikasi Machine Learning')
st.title('Prediksi Potensi Serangan/ Gangguan Jantung')

# Menambah subheader

st.subheader('Beberapa catatan dan informasi')

st.markdown("""
-	Aplikasi ini adalah prediksi menggunakan machine learning mengacu kepada dataset yang diperoleh dari 918 oservasi. 
    Source : https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
-	Dari pengujian beberapa model, dipilih Model Gradien Boosting Classifier, dimana setelah dilakukan proses Model training dengan Cross Validation dan Hyperparameter Tuning, nilai ketepatan prediksi meningkat dari 0,8 menjadi 0,9.
-	Penjelasan dari data-data yang perlu diisi ada di bawah.
-	Untuk masing-masing parameter maupun hasil prediksi perlu konsultasi dengan tenaga kesehatan. 
""")


st.write("""
Ches Pain Type
1.	TA: Typical Angina (Angina Tipikal)
Deskripsi: Nyeri dada yang khas atau klasik yang umumnya disebabkan oleh iskemia miokard (kurangnya pasokan darah ke otot jantung).
Karakteristik: Biasanya dipicu oleh aktivitas fisik atau stres emosional, dan dapat diatasi dengan istirahat atau nitrat.

2.	ATA: Atypical Angina (Angina Atipikal)
Deskripsi: Nyeri dada yang memiliki karakteristik yang tidak sepenuhnya sesuai dengan angina tipikal.
Karakteristik: Mungkin lebih sulit diidentifikasi atau memiliki pola yang tidak khas. Gejala dapat muncul dengan aktivitas fisik atau stres, tetapi tidak selalu.

3.	NAP: Non-Anginal Pain (Nyeri Non-Anginal)
Deskripsi: Nyeri dada yang tidak disebabkan oleh iskemia miokard.
Karakteristik: Meskipun dapat menjadi rasa tidak nyaman di dada, tetapi tidak terkait dengan kurangnya aliran darah ke otot jantung. Penyebabnya dapat beragam, seperti gangguan otot, masalah gastrointestinal, atau faktor lain.

4.	ASY: Asymptomatic (Asimtomatik)
Deskripsi: Tidak ada gejala nyeri dada yang dirasakan.
Karakteristik: Tidak ada keluhan nyeri dada atau ketidaknyamanan di daerah jantung. Ini mungkin terjadi pada individu yang tidak merasakan gejala jantung meskipun memiliki kondisi tertentu.

Resting Blood Pressure (Tekanan Darah Istirahat) adalah tekanan darah => Sistolik

MaxHR atau Maksimum Heart Rate (Denyut Jantung Maksimum)=> Dapat diperoleh saat test treadmill.

ExerciseAngina => sakit dada yang terasa saat Latihan fisik.

Oldpeak => didapat dari EKG / Treadmill test

ST Slop => didapat dari EKG / ECG

RestingECG => didapat dari EKG / ECG

""")

st.markdown("""
pertanyaan, masukan, saran dan kiritik bisa disampaikan via
email : mustakimsr@gmail.com
pesan di IG : mustakim_way

""")

# Load model
my_model = pickle.load(open('heart_disease3.pkl', 'rb'))


# Menulis text (ukuran kecil)
st.subheader('Silahkan Masukkan Data Anda')

# Baris Pertama 1,2
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        ST_Slope = st.selectbox('ST_Slop',['Up','Down','Flat'])
    with col2:
        Cholesterol = st.number_input('Cholesterol', value=0)

#Baris Kedua 3,4
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        MaxHR = st.number_input('MaxHR', value=0)
    with col2:
        Oldpeak = st.number_input('Oldpeak', min_value=-10.0, max_value=50.0, value=-10.0, step=0.1)

# Baris Ketiga 5,6
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input('Age', value=0)
    with col2:
        RestingBP = st.number_input('RestingBP', value=0)
        

# Baris Keempat 7,8
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        Sex = st.selectbox('Sex', ['F','M'])
    with col2:
        ChestPainType = st.selectbox('ChestPainType',['ASY','ATA','NAP','TA'])
        

# Baris Kelima 9,10
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        FastingBS = st.selectbox('Fasting BS, 0 = tidak puasa, 1 = puasa',['0','1'])
    with col2:
        ExerciseAngina = st.selectbox('ExerciseAngina', ['N','Y'])
        
# Baris Keenam 11
with st.container():
        RestingECG = st.selectbox('RestingECG', ['Normal','ST','LVH'])
    
#code untuk prediksi
heart_disease =[]


#membuat tombol prediksi

if st.button('Tes Prediksi Potensi Serangan Jantung'):
    data_list=[]
    
    data_list.append([Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope])
    columns = ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']
    df = pd.DataFrame(data_list, columns=columns)
    
        
    # Menampilkan DataFrame dalam bentuk tabel
    st.write('Tabel Tampilan Data:')
    
    st.table(df.reset_index(drop=True))
        
    heart_disease = my_model.predict(df)
    if (heart_disease[0] == 1):
        heart_disease = 'Berpotensi Serangan Jantung'
    else :
        heart_disease = 'Tidak Berpotensi Serangan Jantung'
        
    st.success(heart_disease)