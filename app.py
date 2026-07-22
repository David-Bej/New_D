from flask import Flask, render_template
import matplotlib.pyplot as plt
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def index():
    # Example data for plotting
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.figure()
    plt.plot(x, y)
    plt.title('Sine Wave')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    
    # Save the plot to a static file
    img_path = os.path.join('static', 'plot.png')
    plt.savefig(img_path)
    plt.close()  # Close the figure to prevent display in notebook

    return render_template('index.html', plot_url=img_path)

if __name__ == '__main__':
    app.run(debug=True)

