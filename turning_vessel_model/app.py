from flask import Flask
from flask_cors import CORS
import numpy as np
from FE import predict
from vessel import Vessel

app = Flask(__name__)
CORS(app)


@app.route("/sim")
def hello_world():
    xtab_FE, ttab_FE = predict(
        t_tot=20,
        t=0,
        dt=0.1,
        u=np.array([[1], [-0.01]], dtype=float),
        vs=Vessel())

    new_dict = {}

    for i in range(len(ttab_FE)):
        inner_dict = {}
        inner_dict['x'] = xtab_FE[i][0][0]
        inner_dict['y'] = xtab_FE[i][1][0]
        inner_dict['psi'] = xtab_FE[i][2][0]
        inner_dict['u'] = xtab_FE[i][3][0]
        inner_dict['c'] = xtab_FE[i][4][0]
        inner_dict['r'] = xtab_FE[i][5][0]
        inner_dict['t'] = round(ttab_FE[i], 1)
        new_dict[round(ttab_FE[i], 1)] = inner_dict

    return (new_dict)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4999, debug=True)
