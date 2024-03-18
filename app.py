from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and DataFrame
pipe = pickle.load(open('pipe.pkl', 'rb'))
data = pickle.load(open('data.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html', companies=data['Company'].unique(), types=data['TypeName'].unique(),
                           cpus=data['cpu brand'].unique(), gpus=data['Gpu brand'].unique(), os=data['os'].unique())

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        company = request.form['company']
        type = request.form['type']
        ram = int(request.form['ram'])
        weight = float(request.form['weight'])
        touchscreen = request.form['touchscreen']
        ips = request.form['ips']
        screen_size = float(request.form['screen_size'])
        resolution = request.form['resolution']
        cpu = request.form['cpu']
        hdd = int(request.form['hdd'])
        ssd = int(request.form['ssd'])
        gpu = request.form['gpu']
        os = request.form['os']

        # Calculate PPI
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
        if touchscreen == 'Yes':
            touchscreen = 1
        else:
            touchscreen = 0

        if ips == 'Yes':
            ips = 1
        else:
            ips = 0

        # Create query array
        query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
        query = query.reshape(1, 12)

        # Predict the price
        predicted_price = int(np.exp(pipe.predict(query)[0]))

        return render_template('index.html', predicted_price=predicted_price, companies=data['Company'].unique(), types=data['TypeName'].unique(),
                               cpus=data['cpu brand'].unique(), gpus=data['Gpu brand'].unique(), os=data['os'].unique())
        return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)


