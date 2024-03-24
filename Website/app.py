from flask import Flask, render_template, request
# from product import poster  # Assuming poster function is defined in product.py

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/product', methods=['GET', 'POST'])
def product():
    if request.method == 'POST':
        text_input = request.form['text_input']
        # image_location = poster(text_input)  # Call the poster function with text input
        return render_template('product.html', image_location=image_location)
    return render_template('product.html')

if __name__ == '__main__':
    app.run(debug=True)
