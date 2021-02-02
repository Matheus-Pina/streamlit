import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import recall_score




@st.cache
def load():
    df_time = pd.read_csv('dataframe_time.csv')
    df_model = pd.read_csv('Dataframe_model.csv')

    df_time.drop('Unnamed: 0', axis = 1, inplace = True)
    df_model.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis = 1, inplace = True)

    df_time['final_date'] = pd.to_datetime(df_time['final_date'], utc=True)

    data = [df_time, df_model]

    return data


df_time = load()[0]
df_model = load()[1]

st.write('''
#  :thumbsup: :thumbsdown: **Análise de Sentimento**
''')
st.write('''
 
''')
st.write('''
1. Neste projeto, foram extraídos os comentários postados pelos usuários do TripAdvisors sobre o Outback do Parkshopping (Brasília - DF).
Além dos comentários, também foram coletadas as datas das postagens e as avaliações. Com isso, foi possível analisar os dados à luz das 
datas de postagens, bem como analisar a percepção dos clientes por meio de suas avaliações.

2. Por fim, utilizando os comentários, foi criado um modelo de *machine learning* para análise de sentimento.
''')


st.sidebar.subheader("**Qual é o seu objetivo?** ")
analise_dados = st.sidebar.checkbox('Análise de Dados')
analise_sentimentos = st.sidebar.checkbox('Análise de Sentimento')


if analise_dados:
    st.write('''
    ## :bar_chart: Análise dos Dados 
    ''')

    dataframe = st.checkbox('Deseja visualizar o Data Frame?')
    if dataframe:
        st.write(df_time)

    ano = st.slider('Escolha o ano desejado', min_value = 2012, max_value = 2020, step = 1)

    # Table Barras Rating x Month
    st.write('''
    ### Analisando as avaliações pelos meses

    No gráfico abaixo, temos no eixo Y as médias das avaliações, e no eixo X, os meses das respectivas médias.
    Assim, podemos analisar se há períodos em que as avaliações são mais positivas ou menos positivas.

    *As avaliações variam entre 1 e 5*
    ''')

    df_year = pd.DataFrame(df_time[df_time.final_date.dt.year == int(ano)].groupby(pd.Grouper(key = 'final_date', freq = 'M'))['rating'].mean())


    X = df_year.index.month_name()
    Y = df_year['rating']

    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))


    ax1.bar(X, Y, color = 'darksalmon')
    ax1.axhline(3, color="r", clip_on=False, lw = 2.5)
    plt.title('Média das Avaliações por mês', fontdict= {'fontsize': 15}, pad = 10)
    plt.ylabel('Average Rating')
    plt.yticks(ticks = list(np.arange(1,5.5,0.5)))
    plt.xticks(rotation=45)
    plt.tight_layout(h_pad=2)

    st.pyplot(fig1)

    

    # Box Plot

    st.write('''
    ### Analisando as avaliações por semana

    No gráfico abaixo, temos no eixo Y as médias das avaliações, e no eixo X, as semanas das respectivas médias.
    Assim, podemos analisar se há períodos em que as avaliações são mais positivas ou menos positivas.

    Um gráfico de boxplot contém algumas informações importantes:

    1. *Os valores são divididos em quartis. Assim, os 1/4 dos dados estão entre o traçado inferior e o 
    início da caixa, 2/4  dos dados estao entre o traçado inferior e o meio da caixa (sinalizado por um traço)
    , que é a mediana. Ou seja, o traço no centro da caixa, divide os dados ao meio.*
    2. *Os outliers, que são valores extremos, são destacados como losangos.*

    *As avaliações variam entre 1 e 5*
    ''')

    filtered_data = df_time[df_time.year == ano]

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))


    sns.boxplot(x = 'name_day' , y = 'rating', data = filtered_data , color = 'lightsalmon' , ax= ax2)

    st.pyplot(fig2)

    # Grafico de Barras Horizontais

    st.write('''
    ### Analisando a quantidade de comentários por semestre

    No gráfico abaixo, as avaliações foram separadas por: valor (1 a 5) e pelos seus respectivos semestres em 
    que foram postadas. Assim, pode-se perceber a quantidade de cada de tipo avaliação ao longo dos semestres.

    ''')

    

    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))
    sns.countplot(y = 'quarter', hue = 'rating',  data = filtered_data, ax = ax3, palette = 'Reds')
    st.pyplot(fig3)

    # Grafico das Quantidades de Palavras

    st.write(''' 
    ### 10 palavras mais frequentes

    Neste gráfico, temos as palavras mais frequentes. O primeiro gráfico mostra as palavras 
    mais frequentes nos comentários positivos, enquanto que o segundo gráfico mostra as 
    dos comentários negativos

    *As avaliações variam de 1 a 5, porém, a quantidade de valores abaixo de 4 era muito pequena,
    de forma que se tornam pouca expressivas estatisticamente. Dessa forma, foi feita uma alteração:*
    
    - *As avaliações com notas 4 ou 5 se tornaram 1*
    - *As avaliações abaixo de 4 (1 a 3) se tornaram 0*

    ''')

    rating_1 = df_model[df_model.rating == 1]['title_text']
    rating_0 = df_model[df_model.rating == 0]['title_text']

    color = ['Pastel2', 'Pastel1']
    ratins = [rating_1, rating_0]

    for item in range(2):
        fign, axn = plt.subplots(1, 1, figsize=(10, 6))
        pd.Series(' '.join([i for i in ratins[item]]).split()).value_counts().head(20).sort_values().plot(kind = 'barh', colormap= color[item])
        
        st.pyplot(fign)

# ---------------------------------------------------------------------------------

elif analise_sentimentos:

    # Modelo

    vectorizer = CountVectorizer()

    vect_feat = vectorizer.fit_transform(df_model['title_text'])
    target = df_model['rating']

    X_train, X_test, y_train, y_test = train_test_split(vect_feat, target,
                                                        stratify = target,
                                                        test_size = 0.1, random_state = 30)

    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    y_predict = nb.predict(X_test)

    
  
    st.write('''
    ## :scroll: Análise de Sentimento
    ''')
    
    dataframe = st.checkbox('Deseja visualizar o Data Frame?')
    
    if dataframe:
        st.write(df_model)

    ano = st.slider('Escolha o ano desejado', min_value = 2012, max_value = 2020, step = 1)

    st.write('''
    ### :cloud: Núvem de Palavras

    Aqui temos mais uma visualização das palavras mais frequentes, e, em uma núvem de palavras,
    a aumento na frequência é perceptível pelo aumento na fonte da palavra. Assim, quanto mais frequente,
    maior é a palavra.
    ''')

    # Nuvem de Palavras

    df_aux = df_model[df_model['year'] == ano]


    word_list = ''

    for w in df_aux['title_text'].tolist():
        word_list = word_list + ''.join(w)


    # gerar uma wordcloud
    mask = np.array(Image.open('country_shape.jpg'))
    wordcloud = WordCloud(background_color="white",
                        width=1600, height=800, mask = mask, colormap = 'Reds'
                        ).generate(word_list)
    
    # mostrar a imagem final

    fig4, ax4 = plt.subplots(figsize=(40,20))
    ax4.imshow(wordcloud, interpolation='bilinear')
    ax4.set_axis_off()
    st.pyplot(fig4)

    # Analise de Sentimento

    st.write('''
    ### Teste o Modelo

    Abaixo, temos dois campos: título e comentário. E, utilizando o modelo criado a partir da análise
    dos dados extraídos do TripAdvisor, pode-ser identificar se o seu comentário seria positivo ou negativo!!!
    
    
    *As avaliações variam de 1 a 5, porém, a quantidade de valores abaixo de 4 era muito pequena,
    de forma que se tornam pouca expressivas estatisticamente. Dessa forma, foi feita uma alteração:*
    
    - *As avaliações com notas 4 ou 5 se tornaram 1*
    - *As avaliações abaixo de 4 (1 a 3) se tornaram 0*

    ''')

    titulo = st.text_input('Título:')
    texto = st.text_input('Comentário: ')

    result = ''


    if st.button('Aplicar'):
        input_text = titulo + ' ' + texto 
        if nb.predict(vectorizer.transform([input_text])) == 1:
            st.write('''
            ### :heart_eyes: Frase Positiva!!
            ''')
            st.subheader('Recall Score: ')
            st.write(recall_score(y_test, y_predict))

        elif nb.predict(vectorizer.transform([input_text])) == 0:
            st.write('''
            ### :cry: Frase Negativa...
            ''')
            st.subheader('Recall Score: ')
            st.write(recall_score(y_test, y_predict))