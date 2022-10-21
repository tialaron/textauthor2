
import allfunctions1

import os
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

className = ["О. Генри", "Стругацкие", "Булгаков", "Клиффорд_Саймак", "Макс Фрай", "Брэдберри"] # Объявляем интересующие нас классы
nClasses = len(className)
test_path11 = '/app/textauthor2/test/'

model01 = load_model('/app/textauthor2/model_author_all.h5')
with open('/app/textauthor2/tokenizer.pickle', 'rb') as handle:
    tokenizer2 = pickle.load(handle)
pictures_path = '/app/textauthor2/pictures/'

st.header('Искусственный Интеллект для определения авторства текста.')

with st.sidebar:
    st.markdown(''' # Содержание:''')
    st.markdown("## [1. Актуальность тематики](#about)", unsafe_allow_html=True)
    st.markdown("## [2. Этапы разработки кейса](#pipeline)", unsafe_allow_html=True)
    st.markdown("## [3. Задача](#task)", unsafe_allow_html=True)
    st.markdown("## [4. Информация о данных](#datadata)", unsafe_allow_html=True)
    st.markdown("## [5. Принцип 'мешок слов' ](#bagofwords)", unsafe_allow_html=True)
    st.markdown("## [6. Демонстрация работы](#neuronwork)", unsafe_allow_html=True)
    st.markdown("## [7. Тест](#voprosi)", unsafe_allow_html=True)

st.header('Актуальность тематики', anchor='about')
st.subheader('Кому будет полезно изучить данную работу ?')
st.write('1.Студентам юридических специальностей. Интересующимся патентоведением.')
st.write('2.Студентам, изучающим принципы информационной безопасности.')
st.write('3.Студентам других специальностей, интересующимся возможностями применением искусственного интеллекта в области изучения текстовых данных.')

st.header('Этапы разработки кейса', anchor='pipeline')
st.image(pictures_path+'realtrack1.jpg',caption='Этапы проектирования кейса')
with st.expander("Общая схема"):
    st.markdown(
        '''
        \n**Этапы:**
        \n1. Создаётся датасет из текстов:
        \nВ данной работе используются тексты шести писателей-фантастов. Собраны произведения братьев Стругацких, О.Генри, Булгакова, Макса Фрая, Рэя Брэдбери и Клиффорда Саймака. 
        \n2. Создается словарь частотности из данных текстов:
        \nКаждое слово встречается определенное количество раз. Подсчитывается частота появления каждого слова и строится их рейтинг. Чем чаще слово повторяется в тексте, тем его рейтинг выше.
        \n3. Обучение сети:
        \nВ процессе обучения сетке демонстрируются отрывки текста и сопоставляется с каждым автором.
        \n4. Демонстрация работы сети:
        \nИз датасета по каждому автору выделяется небольшой отрывок произведения и подается в сеть. Ее задача определить какому автору принадлежит тот или иной фрагмент текста в процентном соотношении. Внимание! В тестовой выборке могут присутствовать фрагменты из текстов, с которыми сеть не знакома. В этом случае она распределит проценты вероятности на всех известных ей авторов.
        ''')

#________________________Задача______________________________
st.header('Задача',anchor='task')
st.write('Необходимо определить какому автору принадлежит тот или иной фрагмент текста. '
         'В данной задаче рассматривается шесть писателей-фантастов и за каждым закреплен датасет из текстов его произведений.')
st.write('Произведения братьев Стругацких, О.Генри, Булгакова, Макса Фрая, Рэя Брэдбери и Клиффорда Саймака весьма разнообразны с точки зрения словесных конструкций и представляют собой отличный разнонаправленный датасет. '
         'Конечно же оригиналы некоторых произведений написаны на английском. Однако мы будем рассматривать экземпляры, переведенные на русский язык.')

st.image(pictures_path+'authorsfiles1.jpg', caption='Здесь показаны файлы с текстами писателей. Это обычные TXT-файлы.')
st.write('Перед тем как начать обучение сетки необходимо разбить весь датасет на  обучающую и тестовую выборки. Таким образом, каждый писатель получает по два файла. В одном тексты для обучающей выборки, в другом - для тестовой.')    

#_________________________Информация о данных___________________________________
st.header('Информация о данных',anchor='datadata')
st.write('В самом начале данные представляют из себя обычные тексты.')
st.image(pictures_path+'authorsfiles2.jpg')
st.write('Однако затем создается так назывемый словарь частотности. Создается он следующим образом. Подсчитывается частота появления слов в тексте и составляется рейтинг этих слов. '
         'Чем чаще употребляется слово, тем оно имеет значение в рейтинге ближе к первому месту. ')
st.image(pictures_path+'authorsfiles3.jpg',caption='каждому слову соответствует свое место в рейтинге.')



col5,col6 = st.columns(2)
with col5:
    one_of_writers = st.radio("Какого автора хотите проверить?",('Айзек Азимов', 'О.Генри', 'Стругацкие', 'Булгаков', 'Клиффорд Саймак', 'Макс Фрай', 'Брэдберри','Ерофеев'))
    if one_of_writers == 'Айзек Азимов':
        st.write('Вы выбрали Азимова.')
        test_path1 = test_path11 + '(Айзек_Азимов) Тестовая_2 вместе.txt'
    elif one_of_writers == 'О.Генри':
        st.write("Вы выбрали О.Генри.")
        test_path1 = test_path11 + '(О. Генри) Тестовая_20 вместе.txt'
    elif one_of_writers == 'Стругацкие':
        st.write("Вы выбрали Стругацких.")
        test_path1 = test_path11 + '(Стругацкие) Тестовая_2 вместе.txt'
    elif one_of_writers == 'Булгаков':
        st.write("Вы выбрали Булгакова.")
        test_path1 = test_path11 + '(Булгаков) Тестовая_2 вместе.txt'
    elif one_of_writers == 'Клиффорд Саймак':
        st.write("Вы выбрали Клиффорда Саймака.")
        test_path1 = test_path11 + '(Клиффорд_Саймак) Тестовая_2 вместе.txt'
    elif one_of_writers == 'Макс Фрай':
        st.write("Вы выбрали Макса Фрая.")
        test_path1 = test_path11 + '(Макс Фрай) Тестовая_2 вместе.txt'
    elif one_of_writers == 'Брэдберри':
        st.write("Вы выбрали Брэдберри.")
        test_path1 = test_path11 + '(Рэй Брэдберри) Тестовая_8 вместе.txt'
    elif one_of_writers == 'Ерофеев':
        st.write("Вы выбрали Ерофеева.")
        test_path1 = test_path11 + 'Erofeev_V_Moskva_Petushki_Тестовая_2 вместе.txt'
with col6:
    newTest = []
    for i in range(nClasses): #Проходим по каждому классу
        newTest.append(allfunctions1.readText(test_path1))
    st.write(newTest[0][:200])

#newTest = []
#for i in range(nClasses): #Проходим по каждому классу
#    newTest.append(allfunctions1.readText(test_path1))

xLen = 1000 #Длина отрезка текста, по которой анализируем, в словах
step = 100 #Шаг разбиения исходного текста на обучающие векторы
testWordIndexes1 = tokenizer2.texts_to_sequences(newTest)  # Проверочные тесты в индексы
wordIndexes = testWordIndexes1

xTest6Classes01 = []               #Здесь будет список из всех классов, каждый размером "кол-во окон в тексте * 20000
xTest6Classes = []                 #Здесь будет список массивов, каждый размером "кол-во окон в тексте * длину окна"(6 по 420*1000)
for wI in wordIndexes:                       #Для каждого тестового текста из последовательности индексов
    sample = (allfunctions1.getSetFromIndexes(wI, xLen, step))  # Тестовая выборка размером "кол-во окон*длину окна"(например, 420*1000)
    xTest6Classes.append(sample)  # Добавляем в список
    xTest6Classes01.append(tokenizer2.sequences_to_matrix(sample))  # Трансформируется в Bag of Words в виде "кол-во окон в тексте * 20000"
xTest6Classes01 = np.array(xTest6Classes01)                     #И добавляется к нашему списку,
xTest6Classes = np.array(xTest6Classes)                     #И добавляется к нашему списку,

xTest = xTest6Classes01

totalSumRec = 0 # Сумма всех правильных ответов
# Проходим по всем классам. А их у нас 6

for i in range(nClasses):
    # Получаем результаты распознавания класса по блокам слов длины xLen
    currPred = model01.predict(xTest[i])
    # Определяем номер распознанного класса для каждохо блока слов длины xLen
    currOut = np.argmax(currPred, axis=1)

    evVal = []
    for j in range(nClasses):
        evVal.append(len(currOut[currOut == j]) / len(xTest[i]))

    totalSumRec += len(currOut[currOut == i])
    recognizedClass = np.argmax(evVal)  # Определяем, какой класс в итоге за какой был распознан

    # Выводим результаты распознавания по текущему классу
    # isRecognized = "Это НЕПРАВИЛЬНЫЙ ответ!"
    # if (recognizedClass == i):
    #  isRecognized = "Это ПРАВИЛЬНЫЙ ответ!"
    str1 = 'Данный текст похож на произведения : ' + className[i] + " " * (11 - len(className[i])) + ' на ' + str(
        int(100 * evVal[i])) + " %"
    # print(str1, " " * (55-len(str1)), isRecognized, sep='')
    st.write(str1, " " * (55 - len(str1)))

sumCount = 0
for i in range(nClasses):
    sumCount += len(xTest[i])
st.write("Средний процент повторимости текста ", int(100 * totalSumRec / sumCount), "%", sep='')
