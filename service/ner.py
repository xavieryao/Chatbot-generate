import service_impl.ner_impl

ner_model_cache = {}


def ner(text, name, raw=False):
    if name not in ner_model_cache:
        loader = service_impl.ner_impl.NERLoader(
            ner_stat='models/' + name + '/ner_mapping.pkl',
            ner_train='dataset/' + name + '/ner/train.dat',
            pre_emb='models/glove/mixed_vec')
        ner_model_cache[name] = loader.load_stat('models/' + name + '/ner.ml')
    model, word_to_id, char_to_id, tag_to_id, id_to_tag = ner_model_cache[name]
    return service_impl.ner_impl.ner(text, model, word_to_id, char_to_id, tag_to_id, id_to_tag, raw=raw)

if __name__ == '__main__':
    import csv
    import json
    with open('service/classify.txt', encoding='utf8') as f:
        lines = []
        for line in f:
            if '\t' not in line:
                continue
            tag, sent = line.split('\t')
            lines.append(sent)

    with open('result.txt', 'w', encoding='utf8') as f:
        for line in lines:
            tags = ner(line, '5c516f95c4952f177b9a70fb', True)
            f.write(json.dumps(tags, ensure_ascii=False))
            f.write('\n')


