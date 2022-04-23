from flask import Flask,render_template,request,redirect,session,Response
from flask.helpers import url_for
import PyPDF2
import numpy as np
import pickle
import re
app=Flask(__name__)
@app.route('/',methods=['POST','GET'])
def home():
    if request.method=='POST':
        f=request.form["myfile"]
        fl = open("c10.pdf","rb")
        reader = PyPDF2.PdfFileReader(fl)
        page1 = reader.getPage(0)
        print(page1)
        pdfData = page1.extractText()
        print(pdfData)
        returned=model(pdfData+"")
        return Response(returned)
    return render_template('home.html')
def model(data):
    encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    vect = pickle.load(open('tfidf.pkl', 'rb'))
    model = pickle.load(open('models.pkl', 'rb'))
    cleaned_text=clean(data)
    cleaned_text = [cleaned_text]
    transformed_data = vect.transform(cleaned_text)
    field = model.predict(transformed_data)
    return encoder.inverse_transform(field)[0]
def clean(resume):
    resume=re.sub('httpS+s*',' ',resume)
    resume=re.sub('RT|cc',' ',resume)
    resume=re.sub('#S+',' ',resume)
    resume=re.sub('@S+','  ',resume)
    resume=re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', resume)
    resume=re.sub(r'[^x00-x7f]',r' ', resume)
    resume=re.sub('s+',' ',resume)
    resume=resume.lower()
    return resume

if __name__=='__main__':
    app.run(port=5000,debug=True)