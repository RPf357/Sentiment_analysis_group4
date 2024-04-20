from flask import Flask, render_template, request
from pyspark.ml import PipelineModel

app = Flask(__name__)

# Load the Spark ML model
model_path = r"C:\UNT AI\Intro to BigData\project_folder\model\spark_ml_model"  
model = PipelineModel.load(model_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get input text from the form
        text = request.form["text"]
        
        # Use the model to make predictions
        prediction = model.transform(text)
        
        # Extract the predicted sentiment
        sentiment = prediction.select("prediction").collect()[0][0]
        
        # Pass the sentiment to the result page
        return render_template("result.html", sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
