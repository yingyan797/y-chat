from flask import Flask, render_template, request, redirect
from utils.processor import Processor
from utils.language import answering

app = Flask(__name__)

class Website:
    def __init__(self, qn="", cx="", segs=[], ev="", ans="") -> None:
        self.question = qn
        self.context = cx
        self.segs = segs
        self.evidence = ev
        self.answer = ans

    def has_request(self):
        return self.question and self.context


site = Website()

@app.route('/loading', methods=['GET', 'POST'])    # main page


@app.route('/', methods=['GET', 'POST'])    # main page
def index():
    fm = request.form
    site.question = fm.get("question")
    site.context = fm.get("context")
    segs = []
    if fm.get("ask") and site.has_request():
        proc = Processor(site.context)
        segs = proc.entailment(site.question)
        for seg in segs:
            if seg[2]:
                site.evidence += proc.text[seg[0]:seg[1]]
        if site.evidence:
            res = answering(site.question, site.evidence)
            if res["score"] > 0.9:
                site.answer = res["answer"]
        if not site.answer:
            site.answer = "Cannot find answer..."
        
    return render_template("index.html", site=site, loading=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)