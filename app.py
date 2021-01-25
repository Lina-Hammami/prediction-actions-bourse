# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 06:28:43 2021
@author: hamem
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# créer une instance de cette classe  
app = Flask(__name__)

# pour charger un modèle  d'apprentissage 
model = pickle.load(open('model.pkl', 'rb'))  

# route par défaut / route racine 
@app.route('/')
def home():
    return render_template('index.html')

# route de prédiction 
@app.route('/predict',methods=['POST'])
def predict():
    '''
     Pour la page de prédiction des actions de bourse 
    '''
    # les entrées de formulaire sont les éléments nécessaires pour la prédiction
    # cette ligne génère un array contenant les valeurs entières des entrées du formulaire 
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]  # encapsuler dans un autre array pour qu'il soit compatible avec le format d'entrée pour le modèle 
    prediction = model.predict(final_features) # effectuer la prédiction à partir du modèle chargé 
    output = round(prediction[0], 2)  # valeur approchée de la valeur de prix de course prédit en 2 chiffres après la virgule
    # plus tard à ajouter comme amélioration le passage de l'accuracy du modèle 
    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
    
    
    
"""
#Pour la visualisation 
    
    import plotly.express as px
    #Global time series chart for daily new cases, recovered, and deaths
    df = global_timeseries #global time series data frame
    fig = px.line(df, x='date', y=['daily new cases','daily new recovered', 'daily new deaths'], title='Global daily new cases')
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()
    
#To put this visualization into the Flask dashboard I must convert the plot into a JSON variable. If I convert the global time series analysis code above into a function, it will look like this (don't forget to import JSON module and use Plotly JSON encoder class):

    def plotly_global_timeseries(global_timeseries):
    df = global_timeseries
    fig = px.line(df, x=’date’, y=[‘daily new cases’,
     ’daily newrecovered’, ‘daily new deaths’], 
      title=’Global daily new cases’)
    fig = fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(width=1500, height=500)
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

#After the function being called, then I can parse the ‘‘plot_json ’’ variable into my .HTM file with this snippet:
    
        
    <div id="plotly-timeseries"></div>
    <script>
    var graph = {{ plot_json | safe }};
    Plotly.plot('plotly-timeseries', graph, {});
    </script>

"""