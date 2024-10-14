# ARMOny Health Bot

ARMOny is an interactive health chatbot that uses Natural Language Processing (NLP) and Multinomial Regression to classify user inputs into various mental health conditions such as Depression, Stress, and Bipolar Disorder. It provides supportive messages and directs users to NHS resources for further assistance.

## Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

## About the Project

The goal of this project was to apply NLP techniques to publicly available datasets from social media platforms (Twitter/Reddit) and develop an interactive chatbot using Streamlit. ARMOny analyzes natural language statements, identifies signs of mental health conditions, and offers guidance by providing:
- Classification of statements into mental health conditions (e.g., Depression, Stress, Bipolar Disorder) using Multinomial Regression.
- Supportive and empathetic messages.
- Links to NHS resources for professional help.

## Features

- **NLP-Based Mental Health Classification:** Classifies user input into mental health categories such as Depression, Stress, Bipolar Disorder, and others using Multinomial Regression.
- **Streamlit App:** Provides an interactive user interface through the Streamlit platform.
- **Supportive Messaging:** Offers empathetic responses based on the user's input.
- **Resource Guidance:** Directs users to NHS resources for further mental health support.
- **Social Media Dataset Analysis:** Leverages social media data from Twitter and Reddit for natural language analysis.

## Usage
- **Run the Streamlit app locally**: **streamlit run app.py**
  
Interact with the chatbot by inputting natural language statements. The bot will classify the statements into mental health conditions using Multinomial Regression and provide supportive feedback.

## Technologies Used
- **Natural Language Processing (NLP)**: For text processing and analysis.
- **Multinomial Regression**: To classify mental health conditions based on user input.

## Datasets: Publicly available social media datasets from Twitter and Reddit.
ARMOny uses the dataset available from Kaggle for training the model. The dataset contains social media data focused on sentiment analysis for mental health conditions.

- **Sentiment Analysis for Mental Health - Kaggle Dataset
This dataset was pre-processed and categorized into different mental health conditions for classification purposes using Multinomial Regression.

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please:

## License
This project is licensed under the MIT License - see the LICENSE file for details.
