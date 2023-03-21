import spacy
from gensim import corpora, models, similarities

# Step 1: Gather dataset of project statements
project_statements = ['Develop a mobile app for language learning',
                      'Create a machine learning model to predict stock prices',
                      'Design a website for an online store']

# Step 2: Preprocess data
nlp = spacy.load('en_core_web_sm')
processed_statements = []
for statement in project_statements:
    doc = nlp(statement.lower())
    tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
    processed_statements.append(tokens)

# Step 3: Train NLP model
dictionary = corpora.Dictionary(processed_statements)
corpus = [dictionary.doc2bow(statement) for statement in processed_statements]
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

# Step 4: Create recommendation engine


def get_similar_projects(statement):
    statement_tokens = [token.text for token in nlp(
        statement.lower()) if not token.is_stop and token.is_alpha]
    statement_bow = dictionary.doc2bow(statement_tokens)
    statement_lsi = lsi[statement_bow]
    index = similarities.MatrixSimilarity(lsi[corpus])
    sims = index[statement_lsi]
    sorted_sims = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    similar_projects = [project_statements[i]
                        for i, score in sorted_sims if score > 0.5]
    return similar_projects

# Step 5: Deploy the model
# You can create a web application or API to deploy the model and allow users to input their project statements and receive recommendations.
