# ntnu-digitial-twin


This project has two main parts:
The turning-vessel-model and the ocean-scene.
The turning-vessel-model simulates and models the ship, and the ocean-scene is the front end visualiser.

To run this project, both python and nodejs are needed. 

The following python packages are needed to run:
numpy
matplotlib
flask
flask-cors

The simulation is written in python and is built on top of a simple flask application and is used to send data to the UI.
The UI is written in React utilising React three Fiber.

Both the react server and the flask server need to be running. 

After install the python requirements, simply run "python app.py" in the turning_vessel_model folder.
Then in a new terminal, run "npm install" from the ocean-scene, then "npm start". This should open a react app in the browser.   

