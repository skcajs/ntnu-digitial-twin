from flask import Flask
import numpy as np
from FE import predict
from vessel import Vessel

app = Flask(__name__)

@app.route("/sim")
def hello_world():
    xtab_FE, ttab_FE = predict(
        t_tot = 20, 
        t = 0, 
        dt = 0.1, 
        u = np.array([[1],[-0.01]], dtype=float), 
        vs= Vessel())
    
    
    new_dict = {}
    for i in range(10):
        inner_dict = {}
        inner_dict['x'] = xtab_FE[i][0][0]
        inner_dict['y'] = xtab_FE[i][1][0]
        inner_dict['psi'] = xtab_FE[i][2][0]
        inner_dict['u'] = xtab_FE[i][3][0]
        inner_dict['c'] = xtab_FE[i][4][0]
        inner_dict['r'] = xtab_FE[i][5][0]
        new_dict[ttab_FE[i]] = inner_dict

    return(new_dict)
