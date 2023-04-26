from flask import Flask, jsonify, make_response, request
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


app = Flask(__name__)


model = SentenceTransformer('nli-distilroberta-base-v2')

actions_groups = [
    {"utter": "check metricbeat <server>", "action": "metricbeat_status"},
    {"utter": "status of metricbeat <server>", "action": "metricbeat_status"},
    {"utter": "metricbeat status <server>", "action": "metricbeat_status"},
    {"utter": "how is metricbeat <server>", "action": "metricbeat_status"},

    {"utter": "restart metricbeat <server>", "action": "metricbeat_restart"},
    {"utter": "resume metricbeat <server>", "action": "metricbeat_restart"},

    {"utter": "any http traffic in clap DC7", "action": "clap_traffic"},

    {"utter": "how is current utilization", "action": "clap_utilization"},
    {"utter": "check utilization", "action": "clap_utilization"},
    {"utter": "utilization status", "action": "clap_utilization"},

    {"utter": "clap service healthiness", "action": "clap_health"},
    {"utter": "clap check", "action": "clap_health"}
]


actions_catalog = []
for act in actions_groups:
    actions_catalog.append(act['utter'])

print(actions_catalog)
"""
actions_catalog = [
    "check metricbeat <server>",
    "status of metricbeat <server>",
    "metricbeat status <server>",
    "how is metricbeat <server>",
    "restart metricbeat <server>",
    "resume metricbeat <server>",
    "any http traffic in clap DC7",
    "how is current utilization",
    "check utilization",
    "utilization status",
    "clap service healthiness",
    "clap check"
]
"""


sentence_embeddings = model.encode(actions_catalog)


@app.route("/")
def status():
    return make_response("OK", 200)

@app.route("/sent_sim", methods=['POST','GET'])
def getSimilarity():
    data = request.get_json()
    question = data['prompt']

    print("question: " + question)

    # test
    # question = ["how is utilization"]
    question = [question]
    question_embedding = model.encode(question)

    print("check point 1")

    similarity_score = cosine_similarity(
        question_embedding,
        sentence_embeddings
    ).flatten()

    print("check point 2")
    print(similarity_score)

    index = np.argmax(similarity_score)

    print("check point 3: " + str(index))
    print("check point 4: " + actions_catalog[index])
    # np.float32(similarity_score[index])

    return {"data": actions_groups[index]['utter'], "action": actions_groups[index]['action'], "score": float(similarity_score[index]) }


if __name__ =="__main__":
    app.run(debug=True,port=7700)