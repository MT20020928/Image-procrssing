from flask import Flask, request, jsonify,render_template

# STEP 1: Import the necessary modules.
from mediapipe.tasks import python
from mediapipe.tasks.python import text


app = Flask(__name__)


@app.route('/', methods=["POST","GET"])  
def success():  
    if request.method == 'POST':  
        INPUT_TEXT = request.form["nm"]
        # Define the input text that you wants the model to classify.
        #INPUT_TEXT = "等待的商品特價中，提供不少優惠折扣"
        #INPUT_TEXT = "商品收到後都破損了，服務人員態度不好，無法溝通"

        # STEP 2: Create an TextClassifier object.
        base_options = python.BaseOptions(model_asset_path="text_classifier.tflite")
        options = text.TextClassifierOptions(base_options=base_options)
        classifier = text.TextClassifier.create_from_options(options)

        # STEP 3: Classify the input text.
        classification_result = classifier.classify(INPUT_TEXT)

        # STEP 4: Process the classification result. In this case, print out the most likely category.
        top_category = classification_result.classifications[0].categories[0]
        print(f'{top_category.category_name} ({top_category.score:.2f})')
        if top_category.category_name=='positive':
             return render_template("index.html",name=top_category.category_name, score=top_category.score, image='yes.jpg')
        if top_category.category_name=='negative':
             return render_template("index.html",name=top_category.category_name, score=top_category.score,image='no.jpg')    
        
    else:
        return render_template("index.html",name='', score='') 

if __name__ == '__main__':
    app.run()