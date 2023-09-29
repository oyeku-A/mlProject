from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import InputData,PredictPipeline

app=Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    form_data = {}
    if request.method=='GET':
        return render_template('home.html')
    else:
        form_data = request.form.to_dict()
        data=InputData(**form_data)
        data_df=data.to_dataframe()
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(data_df)
        return render_template('home.html',results=results[0])


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)        
