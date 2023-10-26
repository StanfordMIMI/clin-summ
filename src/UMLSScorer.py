from quickumls import QuickUMLS
import spacy
import torch.nn

if not spacy.util.is_package("en_core_web_sm"):
    # Download and install the 'en_core_web_sm' model
    spacy.cli.download("en_core_web_sm")

SEMANTICS = ['T017', 'T029', 'T023', 'T030', 'T031', 'T022', 'T025', 'T026', 'T018', 'T021', 'T024', 'T116', 'T195',
             'T123', 'T122', 'T103', 'T120', 'T104', 'T200', 'T196', 'T126', 'T131', 'T125', 'T129', 'T130', 'T197',
             'T114', 'T109', 'T121', 'T192', 'T127', 'T203', 'T074', 'T075', 'T020', 'T190', 'T049', 'T019', 'T047',
             'T050', 'T033', 'T037', 'T048', 'T191', 'T046', 'T184', 'T087', 'T088', 'T028', 'T085', 'T086', 'T038',
             'T069', 'T068', 'T034', 'T070', 'T067', 'T043', 'T201', 'T045', 'T041', 'T044', 'T032', 'T040', 'T042',
             'T039']


class UMLSScorer(torch.nn.Module):

    def __init__(self, use_umls=True,
                 quickumls_fp='/home/vanveen/quickumls/' # path where QuickUMLS is installed
                 ):
        super().__init__()
        self.quickumls_fp = quickumls_fp
        self.WINDOW_SIZE = 5
        self.ENCODING = "utf-8"
        self.SEMANTICS = SEMANTICS
        self.use_umls = use_umls
        self.nlp = spacy.load("en_ner_bc5cdr_md")
        self.matcher = QuickUMLS(self.quickumls_fp, window=self.WINDOW_SIZE, threshold=1,
                                 accepted_semtypes=self.SEMANTICS)

    def get_matches(self, text):
        concepts = {}
        cui_list = []
        if self.use_umls:
            matches = self.matcher.match(text, ignore_syntax=True)
            for match in matches:
                for m in match:
                    if m['cui'] not in concepts.get(m['term'], []):
                        concepts[m['term']] = concepts.get(m['term'], []) + [m['cui']]
                        cui_list.append(m['cui'])
        else:
            doc = self.nlp(text)
            for ent in doc.ents:
                key = (ent.text.lower(), ent.label_)
                if ent.text not in concepts.get(key, []):
                    concepts[key] = concepts.get(key, []) + [ent.text]
                    cui_list.append(ent.text)
        return concepts, cui_list

    def umls_score_individual(self, reference, prediction):
        true_concept, true_cuis = self.get_matches(reference)
        pred_concept, pred_cuis = self.get_matches(prediction)
        try:
            num_t = 0
            for key in true_concept:
                for cui in true_concept[key]:
                    if cui in pred_cuis:
                        num_t += 1
                        break
            precision = num_t * 1.0 / len(pred_concept.keys())
            recall = num_t * 1.0 / len(true_concept.keys())
            F1 = 2 * (precision * recall) / (precision + recall)
            return F1
        except ZeroDivisionError:
            return 0

    def forward(self, reference, prediction):
        return self.umls_score_individual(reference, prediction)

    def umls_score_group(self, references, predictions):
        return [self.umls_score_individual(reference, prediction) for reference, prediction in
                zip(references, predictions)]

