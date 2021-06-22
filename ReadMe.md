# DAEM
Source code, datasets and experimental results for submission "_Deep Entity Matching with Adversarial Active Learning_".

## Dependencies
* Python 3.8.5
* Python libraries: see requirements.txt
* Dataset and pre-trained word embeddings: please use `download.sh` to download

## Datasets (from [DeepMatcher](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md#preprocessed-data))
We apply DAEM on seven benchmark datasets.

## Project structure
<table>
    <tr>
        <th>File</th><th>Description</th>
    </tr>
    <tr>
        <td>daem</td><td>Code of DAEM</td>
    </tr>
    <tr>
        <td>scripts/entity_matching.py</td><td>Code to reproduce the deep entity matching experiment and ablation study</td>
    </tr>
    <tr>
        <td>scripts/adversarial_active_learning.py</td><td>Code to reproduce the active entity matching experiment</td>
    </tr>
    <tr>
        <td>scripts/inter_attribute_completion.py</td><td>Code to reproduce the inter-attribute completion experiment</td>
    </tr>
    <tr>
        <td>scripts/adversarial_learning.py</td><td>Code to reproduce the adversarial learning experiment</td>
    </tr>
    <tr>
        <td>scripts/entity_signature.py</td><td>Code to reproduce the entity signature experiment</td>
    </tr>
    <tr>
        <td>scripts/dynamic_blocking.py</td><td>Code to reproduce the dynamic blocking experiment</td>
    </tr>
</table>
