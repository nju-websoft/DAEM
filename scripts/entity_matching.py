from daem.models import pure_plus
task_names = ['Amazon-Google', 'DBLP-ACM', 'DBLP-GoogleScholar', 'iTunes-Amazon', 
                  'Walmart-Amazon', 'Fodors-Zagats', 'Beer']

for task_name in task_names:
    pure_plus.test_model(task_name)

pure_plus.with_token_weight = False
for task_name in task_names:
    pure_plus.test_model(task_name)
pure_plus.with_token_weight = True

pure_plus.with_transfomer = False
for task_name in task_names:
    pure_plus.test_model(task_name)
pure_plus.with_transfomer = True

pure_plus.with_similarity_vectors = 0
for task_name in task_names:
    pure_plus.test_model(task_name)
pure_plus.with_similarity_vectors = 1

pure_plus.with_difference_vectors = 0
for task_name in task_names:
    pure_plus.test_model(task_name)
pure_plus.with_difference_vectors = 1