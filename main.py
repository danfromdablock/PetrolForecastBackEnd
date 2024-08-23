from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from werkzeug.utils import secure_filename


class ForecastSingleDecline:
    """
    Forecasts single decline production using Exponential, Hyperbolic, and Harmonic models.

    Attributes:
        period (int): Number of periods for the forecast.
        rate (float): Initial production rate.
        decline_rate (float): Decline rate for exponential and hyperbolic models.
        b_factor (float): b-factor for hyperbolic model.
        econ_limit (float): Economic limit.
    """

    def __init__(self, period, rate, decline_rate, b_factor, econ_limit):
        self.period = period
        self.rate = rate
        self.decline_rate = decline_rate
        self.b_factor = b_factor
        self.econ_limit = econ_limit

    def exp_forecast(self):
        """Calculates the exponential forecast."""
        t = np.arange(0, self.period)
        return self.rate * np.exp(-self.decline_rate * t)

    def hyp_forecast(self):
        """Calculates the hyperbolic forecast."""
        t = np.arange(0, self.period)
        return self.rate / (1 + self.b_factor * self.decline_rate * t) ** (1 / self.b_factor)

    def har_forecast(self):
        """Calculates the harmonic forecast."""
        t = np.arange(0, self.period)
        return self.rate / (1 + self.decline_rate * t)

    def cum_forecast(self, forecast_function):
        """Calculates the cumulative forecast."""
        forecast = forecast_function()
        return np.cumsum(forecast)

    def get_forecasts(self):
        """Returns forecasts for all models."""
        exp_forecast = self.exp_forecast()
        hyp_forecast = self.hyp_forecast()
        har_forecast = self.har_forecast()

        exp_cum = self.cum_forecast(self.exp_forecast)
        hyp_cum = self.cum_forecast(self.hyp_forecast)
        har_cum = self.cum_forecast(self.har_forecast)

        return {
            'exp_forecast': exp_forecast,
            'hyp_forecast': hyp_forecast,
            'har_forecast': har_forecast,
            'exp_cum': exp_cum,
            'hyp_cum': hyp_cum,
            'har_cum': har_cum
        }


class ForecastTwoPeriods:
    """
    Forecasts production using two periods with different decline rates and b-factors.

    Attributes:
        period (int): Number of periods for the forecast.
        rate (float): Initial production rate.
        d1 (float): Decline rate for the first period.
        d2 (float): Decline rate for the second period.
        b1 (float): b-factor for the first period.
        p1 (int): Transition period.
        b2 (float): b-factor for the second period.
        econ_limit (float): Economic limit.
    """

    def __init__(self, period, rate, d1, d2, b1, p1, b2, econ_limit):
        self.period = period
        self.rate = rate
        self.d1 = d1
        self.d2 = d2
        self.b1 = b1
        self.p1 = p1
        self.b2 = b2
        self.econ_limit = econ_limit

    def exp_forecast(self):
        """Calculates the exponential forecast for two periods."""
        t = np.arange(0, self.period)
        forecast = np.zeros(self.period)
        for i in range(self.period):
            if i < self.p1:
                forecast[i] = self.rate * np.exp(-self.d1 * i)
            else:
                forecast[i] = self.rate * np.exp(-self.d1 * self.p1) * np.exp(-self.d2 * (i - self.p1))
        return forecast

    def hyp_forecast(self):
        """Calculates the hyperbolic forecast for two periods."""
        t = np.arange(0, self.period)
        forecast = np.zeros(self.period)
        for i in range(self.period):
            if i < self.p1:
                forecast[i] = self.rate / (1 + self.b1 * self.d1 * i) ** (1 / self.b1)
            else:
                forecast[i] = (self.rate / (1 + self.b1 * self.d1 * self.p1) ** (1 / self.b1)) / (
                            1 + self.b2 * self.d2 * (i - self.p1)) ** (1 / self.b2)
        return forecast

    def har_forecast(self):
        """Calculates the harmonic forecast for two periods."""
        t = np.arange(0, self.period)
        forecast = np.zeros(self.period)
        for i in range(self.period):
            if i < self.p1:
                forecast[i] = self.rate / (1 + self.d1 * i)
            else:
                forecast[i] = self.rate / (1 + self.d1 * self.p1) / (1 + self.d2 * (i - self.p1))
        return forecast

    def cum_forecast(self, forecast_function):
        """Calculates the cumulative forecast."""
        forecast = forecast_function()
        return np.cumsum(forecast)

    def get_forecasts(self):
        """Returns forecasts for all models."""
        exp_forecast = self.exp_forecast()
        hyp_forecast = self.hyp_forecast()
        har_forecast = self.har_forecast()

        exp_cum = self.cum_forecast(self.exp_forecast)
        hyp_cum = self.cum_forecast(self.hyp_forecast)
        har_cum = self.cum_forecast(self.har_forecast)

        return {
            'exp_forecast': exp_forecast,
            'hyp_forecast': hyp_forecast,
            'har_forecast': har_forecast,
            'exp_cum': exp_cum,
            'hyp_cum': hyp_cum,
            'har_cum': har_cum
        }


def plot_forecast(forecasts, econ_limit):
    """
    Plots the forecasts using both Matplotlib and Plotly for comparison.

    Args:
        forecasts (dict): Forecast data containing exponential, hyperbolic, and harmonic forecasts.
        econ_limit (float): Economic limit line to be plotted.
    """
    periods = np.arange(len(forecasts['exp_forecast']))

    # Matplotlib Plot
    plt.figure(figsize=(12, 6))
    plt.plot(periods, forecasts['exp_forecast'], label='Exponential')
    plt.plot(periods, forecasts['hyp_forecast'], label='Hyperbolic')
    plt.plot(periods, forecasts['har_forecast'], label='Harmonic')
    plt.axhline(y=econ_limit, color='r', linestyle='--', label='Economic Limit')
    plt.xlabel('Time Period')
    plt.ylabel('Production Rate')
    plt.title('Forecasting (Matplotlib)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotly Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=periods, y=forecasts['exp_forecast'], mode='lines', name='Exponential'))
    fig.add_trace(go.Scatter(x=periods, y=forecasts['hyp_forecast'], mode='lines', name='Hyperbolic'))
    fig.add_trace(go.Scatter(x=periods, y=forecasts['har_forecast'], mode='lines', name='Harmonic'))
    fig.add_trace(go.Scatter(x=[0, max(periods)], y=[econ_limit, econ_limit], mode='lines', name='Economic Limit',
                             line=dict(color='red', dash='dash')))
    fig.update_layout(title='Forecasting (Plotly)', xaxis_title='Time Period', yaxis_title='Production Rate',
                      legend_title='Forecast Models')
    fig.show()


def save_results(forecasts, filename='forecasts.xlsx'):
    """
    Saves the forecast results to an Excel file.

    Args:
        forecasts (dict): Forecast data containing exponential, hyperbolic, and harmonic forecasts.
        filename (str): The filename for the Excel file.
    """
    df = pd.DataFrame({
        'Period': np.arange(len(forecasts['exp_forecast'])),
        'Exp Forecast': forecasts['exp_forecast'],
        'Hyp Forecast': forecasts['hyp_forecast'],
        'Har Forecast': forecasts['har_forecast'],
        'Exp Cumulative': forecasts['exp_cum'],
        'Hyp Cumulative': forecasts['hyp_cum'],
        'Har Cumulative': forecasts['har_cum']
    })
    df.to_excel(filename, index=False)


def main():
    """
    Main function to run examples and demonstrate the forecasting tool.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Petroleum Forecasting Tool')
    parser.add_argument('--model', choices=['single', 'two_periods'], required=True, help='Forecast model to use.')
    parser.add_argument('--period', type=int, default=12, help='Number of periods for the forecast.')
    parser.add_argument('--rate', type=float, default=1000, help='Initial production rate.')
    parser.add_argument('--decline_rate', type=float, default=0.05, help='Decline rate for the exponential model.')
    parser.add_argument('--b_factor', type=float, default=0.5, help='b-factor for the hyperbolic model.')
    parser.add_argument('--econ_limit', type=float, default=10, help='Economic limit.')
    parser.add_argument('--d1', type=float, help='Decline rate for the first period (for two periods model).')
    parser.add_argument('--d2', type=float, help='Decline rate for the second period (for two periods model).')
    parser.add_argument('--b1', type=float, help='b-factor for the first period (for two periods model).')
    parser.add_argument('--p1', type=int, help='Transition period (for two periods model).')
    parser.add_argument('--b2', type=float, help='b-factor for the second period (for two periods model).')
    parser.add_argument('--filename', type=str, default='forecasts.xlsx', help='Filename to save the results.')

    args = parser.parse_args()

    if args.model == 'single':
        forecaster = ForecastSingleDecline(
            period=args.period,
            rate=args.rate,
            decline_rate=args.decline_rate,
            b_factor=args.b_factor,
            econ_limit=args.econ_limit
        )
    elif args.model == 'two_periods':
        if None in [args.d1, args.d2, args.b1, args.p1, args.b2]:
            raise ValueError('For the two_periods model, d1, d2, b1, p1, and b2 must be provided.')
        forecaster = ForecastTwoPeriods(
            period=args.period,
            rate=args.rate,
            d1=args.d1,
            d2=args.d2,
            b1=args.b1,
            p1=args.p1,
            b2=args.b2,
            econ_limit=args.econ_limit
        )

    forecasts = forecaster.get_forecasts()

    # Plot the forecasts
    plot_forecast(forecasts, args.econ_limit)

    # Save the results
    save_results(forecasts, args.filename)


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#this plcae the data graph to the front end
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the uploaded file (assuming it's Excel)
        df = pd.read_excel(filepath)
        # You can now use df to get the relevant data for forecasting

        # Example of returning mock forecast data
        forecast_data = {
            'labels': list(df.index),
            'values': list(df['production'])  # Adjust this based on your dataframe structure
        }

        return jsonify({'forecast': forecast_data})

    return jsonify({'error': 'File type not allowed'})


@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.json
    model = data['model']
    period = int(data['period'])
    rate = float(data['rate'])
    decline_rate = float(data['decline_rate'])
    b_factor = float(data['b_factor'])
    econ_limit = float(data['econ_limit'])

    if model == 'single':
        forecaster = ForecastSingleDecline(period, rate, decline_rate, b_factor, econ_limit)
    elif model == 'two_periods':
        d1 = float(data['d1'])
        d2 = float(data['d2'])
        b1 = float(data['b1'])
        p1 = int(data['p1'])
        b2 = float(data['b2'])
        forecaster = ForecastTwoPeriods(period, rate, d1, d2, b1, p1, b2, econ_limit)

    forecasts = forecaster.get_forecasts()
    graph_html = plot_forecast(forecasts, econ_limit)
    return jsonify({'graph': graph_html})


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)