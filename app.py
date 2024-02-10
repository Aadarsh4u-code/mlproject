from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import (CustomData, PredictPipeline)

application_name = Flask(__name__)
app = application_name


# Route for a Home Page
@app.route('/')
def index():
    return render_template('index.html', title='Students Progress')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html', title='Students Predict')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=int(request.form.get('writing_score')),
            writing_score=int(request.form.get('reading_score'))
        )
        print('data', data)
        pred_df = data.get_data_as_data_frame()
        print('pred_df', pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print('results', results)
        return render_template('home.html', title='Students Progress Predict', results=results[0])


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
