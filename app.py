[label app.py]

from flask import json, request, Flask
from flask_restful import Api, Resource

from model import ClassificationModel
from utils import split_datasets

app = Flask(__name__)
api = Api(app)

#@app.route('/', methods = ['POST', 'GET'])
class Model(Resource):
    def get(self):
        
        model = ClassificationModel()

        jsondata = request.get_json(force=True)
        if not isinstance(jsondata, dict):
            data = json.loads(jsondata)
        else:
            data = jsondata

        
        #df_fall, df_skin, df_readmission = split_datasets(data)
        
        model.load_data(data)
        model.classify()

        result = model.get_predictions()
        

        return json.dumps(result)

api.add_resource(Model, '/')

if __name__ == '__main__':
    app.run(debug=True)
