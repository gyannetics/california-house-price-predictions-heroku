from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit(): 
    # HTML to PYTHON
    if request.method == 'POST':
        name = request.form['username']

    # PYTHON TO HTML
    return render_template('submit.html', n=name)

if __name__ == '__main__':
    app.run(debug=True)