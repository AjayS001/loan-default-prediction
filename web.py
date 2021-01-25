from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd


model=pickle.load(open('model.pkl','rb'))


app = Flask(__name__)
@app.route("/")
def index():
    return render_template('index.html')
@app.route("/about")
def about():
    return render_template('about.html')
@app.route("/contact")
def contact():
    return render_template('contact.html')
@app.route("/services")
def services():
    return render_template('services.html')
@app.route("/defaultPredict")
def defaultPredict():
    return render_template('defaultPredict.html')
@app.route("/result", methods=["GET","POST"])
def result():
    SK_ID_CURR=request.form['SK_ID_CURR']
    CNT_CHILDREN=request.form['CNT_CHILDREN']
    AMT_INCOME_TOTAL=request.form['AMT_INCOME_TOTAL']
    AMT_GOODS_PRICE_x=request.form['AMT_GOODS_PRICE_x']
    REGION_POPULATION_RELATIVE=request.form['REGION_POPULATION_RELATIVE']
    DAYS_EMPLOYED=request.form['DAYS_EMPLOYED']
    DAYS_REGISTRATION=request.form['DAYS_REGISTRATION']
    DAYS_ID_PUBLISH=request.form['DAYS_ID_PUBLISH']
    REG_REGION_NOT_LIVE_REGION=request.form['REG_REGION_NOT_LIVE_REGION']
    LIVE_REGION_NOT_WORK_REGION=request.form['LIVE_REGION_NOT_WORK_REGION']
    REG_CITY_NOT_LIVE_CITY=request.form['REG_CITY_NOT_LIVE_CITY']
    REG_CITY_NOT_WORK_CITY=request.form['REG_CITY_NOT_WORK_CITY']
    LIVE_CITY_NOT_WORK_CITY=request.form['LIVE_CITY_NOT_WORK_CITY']
    DEF_30_CNT_SOCIAL_CIRCLE=request.form['DEF_30_CNT_SOCIAL_CIRCLE']
    OBS_60_CNT_SOCIAL_CIRCLE=request.form['OBS_60_CNT_SOCIAL_CIRCLE']
    DEF_60_CNT_SOCIAL_CIRCLE=request.form['DEF_60_CNT_SOCIAL_CIRCLE']
    DAYS_LAST_PHONE_CHANGE=request.form['DAYS_LAST_PHONE_CHANGE']
    FLAG_DOCUMENT_3=request.form['FLAG_DOCUMENT_3']
    FLAG_DOCUMENT_6=request.form['FLAG_DOCUMENT_6']
    FLAG_DOCUMENT_8=request.form['FLAG_DOCUMENT_8']
    AMT_REQ_CREDIT_BUREAU_HOUR=request.form['AMT_REQ_CREDIT_BUREAU_HOUR']
    AMT_REQ_CREDIT_BUREAU_DAY=request.form['AMT_REQ_CREDIT_BUREAU_DAY']
    AMT_REQ_CREDIT_BUREAU_WEEK=request.form['AMT_REQ_CREDIT_BUREAU_WEEK']
    AMT_REQ_CREDIT_BUREAU_MON=request.form['AMT_REQ_CREDIT_BUREAU_MON']
    AMT_REQ_CREDIT_BUREAU_QRT=request.form['AMT_REQ_CREDIT_BUREAU_QRT']
    AMT_REQ_CREDIT_BUREAU_YEAR=request.form['AMT_REQ_CREDIT_BUREAU_YEAR']
    AMT_ANNUITY_y=request.form['AMT_ANNUITY_y']
    AMT_APPLICATION=request.form['AMT_APPLICATION']
    SELLERPLACE_AREA=request.form['SELLERPLACE_AREA']
    CNT_PAYMENT=request.form['CNT_PAYMENT']
    DAYS_DECISION=request.form['DAYS_DECISION']
    arry=np.array([[SK_ID_CURR,CNT_CHILDREN,AMT_INCOME_TOTAL,AMT_GOODS_PRICE_x,REGION_POPULATION_RELATIVE,
                    DAYS_EMPLOYED,DAYS_REGISTRATION,DAYS_ID_PUBLISH,REG_REGION_NOT_LIVE_REGION,
                    LIVE_REGION_NOT_WORK_REGION,REG_CITY_NOT_LIVE_CITY,REG_CITY_NOT_WORK_CITY,
                    LIVE_CITY_NOT_WORK_CITY,DEF_30_CNT_SOCIAL_CIRCLE,OBS_60_CNT_SOCIAL_CIRCLE,
                    DEF_60_CNT_SOCIAL_CIRCLE,DAYS_LAST_PHONE_CHANGE,FLAG_DOCUMENT_3,FLAG_DOCUMENT_6,
                    FLAG_DOCUMENT_8,AMT_REQ_CREDIT_BUREAU_HOUR,AMT_REQ_CREDIT_BUREAU_DAY,AMT_REQ_CREDIT_BUREAU_WEEK,
                    AMT_REQ_CREDIT_BUREAU_MON,AMT_REQ_CREDIT_BUREAU_QRT,AMT_REQ_CREDIT_BUREAU_YEAR,
                    AMT_ANNUITY_y,AMT_APPLICATION,SELLERPLACE_AREA,CNT_PAYMENT,DAYS_DECISION]])
   
   
    
    prediction = model.predict(arry)
    
    Repayer= 'Loan applicant is classified as Repayer'
    Defaulter = 'Loan applicant is classified as Defaulter'
    
    if prediction == 0:
        return render_template('result.html', prediction=Repayer)
    else :
        return render_template('result.html', prediction=Defaulter)
    
    

    return render_template('result.html')

if __name__=="__main__":
    app.run()
