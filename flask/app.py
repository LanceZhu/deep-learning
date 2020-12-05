from flask import Flask, request, render_template
from PIL import Image
import io
from models.dnn import predict as dnn_predict
from models.cnn import predict as cnn_predict
from models.knn import predict as knn_predict
from models.svm import predict as svm_predict
app = Flask(__name__)

@app.route('/mnist')
def index():
  return render_template('index.html')

@app.route('/api/mnist', methods=['POST'])
def mnist_predict():
  file = request.files['img']
  img_bytes = file.read()
  image = Image.open(io.BytesIO(img_bytes)).convert('L')
  image = image.resize((28, 28))
  image.save('number.png')

  dnn_num = dnn_predict(image)
  cnn_num = cnn_predict(image)
  knn_num = knn_predict(image)
  svm_num = svm_predict(image)

  return {
    "dnn": dnn_num,
    "cnn": cnn_num,
    "knn": knn_num,
    "svm": svm_num
  }
