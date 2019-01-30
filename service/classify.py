import service_impl.classify_impl

classify_model_cache = {}

def classify(text, name, raw=False):
    if name not in classify_model_cache:
        loader = service_impl.classify_impl.ClassifyLoader(
            name=name,
            pre_emb='models/glove/mixed_vec')
        classify_model_cache[name] = loader.load_stat('models/' + name + '/classify.ml')
    model, d_word_index = classify_model_cache[name]
    return service_impl.classify_impl.classify(text, model, d_word_index, raw)


# def classify(text, lang):
#     return service_impl.classify_impl.classify(text, model, d_word_index, lang)

if __name__ == '__main__':
    import csv
    import json
    with open('service/aminer_data.csv', encoding='utf8') as f:
        lines = list(csv.reader(f))[1:]
    output = []
    with open('output.txt', 'w') as f:
        for line in lines:
            out = classify(line[0], '5bfe0b04c4952f342f394a42', True)
            output.append(out)
            f.write(str(out))
            f.write('\n')
