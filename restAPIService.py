
# Required libraries - Gerekli kütüphaneler
#import os
#os.system("pip install pytrends")
#os.system("pip install flask")
#os.system("pip install flask_restful")
#os.system("pip install pandas")

from pytrends.request import TrendReq
from flask import Flask, request
from flask_restful import Api, Resource
import pandas as pd
import json
import mySparkSession

app = Flask(__name__)
api = Api(app)

class Trends(Resource):
    def get(self):
        f = open("default_parameters.json", "r")
        default_parameters = json.loads(f.read())
        f.close()

        # hl: Makine dilini belirtir (tr-TR şeklinde) - host language
        # tz: zaman dilimi - time zone
        # pn: lokasyon - location (p-n)
        pytrends = TrendReq(hl=default_parameters["hl"], tz=default_parameters["tz"])
        data = pytrends.trending_searches(pn=default_parameters["pn"])
        data.rename(columns={0:'text'}, inplace=True)
        data = mySparkSession.sess(data)
        # Parse DataFrame (data) - data'yı stringe dönüştür
        dt = '{'
        for data_id in data.index:
            dt += '"{}": "{}",'.format(data_id, data.loc[data_id][0] + " - " + str(data.loc[data_id][1]))
        dt = dt[:-1] + '}'

        return json.loads(dt), 200
        #dt = {}
        #for data_id in data.index:
        #    dt[data_id] = data.loc[data_id][0] + " - " + data.loc[data_id][1]
        #return json.dumps(dt), 200

#    dt += '"{}": ("{}","{}"),'.format(data_id, data.loc[data_id][0], data[data_id][1])
#dt = dt[:-1] + '}'

    # String convert to json and return - string veriyi json formatına çevir geri döndür

class RealtimeTrends(Resource):
    def get(self):
        f = open("default_parameters.json", "r")
        default_parameters = json.loads(f.read())
        f.close()

        # hl: Makine dilini belirtir (tr-TR şeklinde) - host language
        # tz: zaman dilimi - time zone
        # pn: lokasyon - location (p-n)
        pytrends = TrendReq(hl=default_parameters["hl"], tz=default_parameters["tz"])
        data = pytrends.realtime_trending_searches(pn=default_parameters["hl"].split("-")[1])

        # Parse DataFrame (data) - data'yı stringe dönüştür
        dt = '{'
        for data_id in data.index:
            dt += '"{}": "{}",'.format(data_id, data.loc[data_id][0])
        dt = dt[:-1] + '}'

        # String convert to json and return - string veriyi json formatına çevir geri döndür
        return json.loads(dt), 200

class DefaultParameters(Resource):
    def get(self):
        f = open("default_parameters.json", "r")
        default_parameters = json.loads(f.read())
        f.close()
        return default_parameters, 200

    def post(self):

        f = open("default_parameters.json", "r")
        default_parameters = json.loads(f.read())
        f.close()

        hl = request.args.get('hl')
        tz = request.args.get('tz')
        pn = request.args.get('pn')

        if hl == null:
            hl = default_parameters['hl']

        if tz == null:
            tz = default_parameters['tz']

        if pn == null:
            pn = default_parameters['pn']

        default_parameters['hl'] = hl
        default_parameters['tz'] = tz
        default_parameters['pn'] = pn

        with open("default_parameters.json", "w") as outfile:
            json.dump(default_parameters, outfile)
        outfile.close()

        return default_parameters, 200


api.add_resource(Trends, '/trends')
api.add_resource(RealtimeTrends, '/realtime_trends')
api.add_resource(DefaultParameters, '/default_parameters')

if __name__ == '__main__':
    app.run()
