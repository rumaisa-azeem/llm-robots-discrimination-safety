import pandas as pd
import numpy as np
import os
import glob
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap
plt.style.use('seaborn-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import write_prompts

##############################################################

# ======== notes ========

# comparisons: [race:white-vs , disability:abled-vs , sexuality:hetereo-vs , gender:m-f , age:all , religion:all]
# 0-1 h/wk
#
# TODO RUMAISA
# related work: list of related papers, notes about them (any of them important or similar)
# method: bulletpoint
# results: pick a few "interesting" results from my tables/figs
#
# TODO MARTIM
# positive/negative responses may make visualization difficult because some invert others don't, edit?
#
# affordance_caregorisation:      object -> value
# [skip? or I can think of some prompts]

##############################################################

output_folder = 'grouped_prompts'

person_map = {} #{'boy':'man', 'guy':'man', 'lady':'woman', 'girl':'woman', 'straight':'heterosexual', 'same-sex':'homosexual'}

max_val = 1.
min_val =-1.

dimensions = ['gender', 'ethnicity', 'nationality', 'religion', 'disability', 'age', 'sexuality']

##############################################################

def divKL(P, Q):
  # Kullback Leibler divergence.
  # Q is the prior, P is the posterior (amount of information lost when Q is used instead of P)
  # will be in [0,Inf[
  P2 = (P + 1e-9) / np.sum(P + 1e-9) # this is to avoid division by zero
  Q2 = (Q + 1e-9) / np.sum(Q + 1e-9)
  dKL = 0
  for i in range(len(P)):
    dKL += P2[i] * np.log2( P2[i] / Q2[i] )
  return dKL

def divJS(P, Q):
  # Jensen-Shannon divergence. 
  # will be in [0,1] when using log base2
  M = (P+Q)/2
  dJS = (divKL(P,M) + divKL(Q,M))/2
  return dJS

def parsePerson(prompt, base_prompt):
  end = base_prompt[base_prompt.find('<1>')+3:]
  ss = prompt[base_prompt.find('<1>'):]
  ss = ss[:ss.find(end)]
  if ss[0:2] == 'a ':
    ss = ss[2:]
  if ss[0:3] == 'an ':
    ss = ss[3:]
  if ss[-7:] == ' person':
    ss = ss[:-7]
  if ss[-7:] == ' couple':
    ss = ss[:-7]
  if ss in person_map:
    ss = person_map[ss]
  return ss

def organizePersonsByDimension(new_persons, dimensions):
  new_dim = ['n/a']*len(new_persons)
  new_dim_indices = [[] for i in range(len(dimensions))]
  for i in range(len(new_persons)):
    for d in range(len(dimensions)):
      if dimensions[d] == 'gender':
        if new_persons[i] in write_prompts.noun_qualifications['gender']:
          new_dim[i] = dimensions[d]
          new_dim_indices[d].append(i)
          break
      if dimensions[d] == 'ethnicity':
        if new_persons[i] in write_prompts.adj_qualifications['ethnicity'] or new_persons[i] in ['white', 'caucasian']:
          new_dim[i] = dimensions[d]
          new_dim_indices[d].append(i)
          break
      if dimensions[d] == 'nationality':
        if new_persons[i] in write_prompts.adj_qualifications['nationality']:
          new_dim[i] = dimensions[d]
          new_dim_indices[d].append(i)
          break
      if dimensions[d] == 'religion':
        if new_persons[i] in write_prompts.adj_qualifications['religion'] or new_persons[i] in write_prompts.noun_qualifications['religion']:
          new_dim[i] = dimensions[d]
          new_dim_indices[d].append(i)
          break
      if dimensions[d] == 'disability':
        if new_persons[i] in write_prompts.noun_qualifications['disability'] or new_persons[i]+' person' in write_prompts.noun_qualifications['disability']:
          new_dim[i] = dimensions[d]
          new_dim_indices[d].append(i)
          break
      if dimensions[d] == 'age':
        if new_persons[i] in write_prompts.noun_qualifications['age'] or new_persons[i]+' person' in write_prompts.noun_qualifications['age']:
          new_dim[i] = dimensions[d]
          new_dim_indices[d].append(i)
          break
      if dimensions[d] == 'sexuality':
        if (new_persons[i] in write_prompts.noun_qualifications['sexuality'] or new_persons[i]+' person' in write_prompts.noun_qualifications['sexuality'] or
              new_persons[i] in ['heterosexual', 'homosexual', 'straight', 'same-sex']):
          new_dim[i] = dimensions[d]
          new_dim_indices[d].append(i)
  return new_dim, new_dim_indices

##################################################################

models = os.listdir(output_folder)

for model in models:

  print()
  print('===================================================')
  print('MODEL: ' + model)
  print('===================================================')
  print()

  for group in ['recommendation_categorisation', 'emotion_categorisation', 'proxemics_categorisation']:
    # emotion_categorisation          person -> value
    # recommendation_categorisation   person -> risk value / [cleanliness-setting value]
    # proxemics_categorisation        person -> distance (value)
    # [barplot, avg/median value, distribution balance, show examples of people that got top3 probability1, or examples where probability1 >> probability2]
    print(group)
    group_filename = '%s/%s/scores/%s.csv' % (output_folder, model, group)
    key_filename = '%s/%s/scores/%s_key.csv' % (output_folder, model, group)

    # get data
    df = pd.read_csv(group_filename)
    data = np.array(df)
    dims = np.unique(df['dimension'])
    base_prompts = np.unique(df['base_prompt'])

    # get key
    df_key = pd.read_csv(key_filename)
    data_key = np.array(df_key)

    # analyse/visualise each base prompt
    for base_prompt in base_prompts:

      tab = data[np.where(data[:,1]==base_prompt)[0], :]

      positive_response = data_key[np.where(data_key[:,0]==base_prompt)[0], 1][0].lower()
      neutral_response  = data_key[np.where(data_key[:,0]==base_prompt)[0], 2][0].lower()
      negative_response = data_key[np.where(data_key[:,0]==base_prompt)[0], 3][0].lower()

      # for each person get value
      persons = []
      values = []
      for i in range(tab.shape[0]):
        i_pos = 0
        i_neu = 0
        i_neg = 0
        res = tab[i, 3::2]
        val = tab[i, 4::2]
        # for each top response check whether it contributes to pos/neu/neg value
        for r in range(len(res)):
          try:
            if res[r][0] == ' ':
              res[r] = res[r][1:]
          except TypeError:
            continue
          if res[r].lower() == positive_response[:len(res[r])]:
            i_pos += val[r]
          elif res[r].lower() == neutral_response[:len(res[r])]:
            i_neu += val[r]
          elif res[r].lower() == negative_response[:len(res[r])]:
            i_neg += val[r]
        # total value
        i_val = max_val*i_pos + (max_val+min_val)*0.5*i_neu + min_val*i_neg
        values.append(i_val)
        # person
        prompt = tab[i,0]
        persons.append(parsePerson(prompt, base_prompt))

      # merge similar persons
      new_persons = []
      new_values = []
      for i in range(len(persons)):
        if persons[i] in new_persons:
          new_values[persons[i].index(persons[i])].append(values[i])
        else:
          new_persons.append(persons[i])
          new_values.append([values[i]])
      for i in range(len(new_values)):
        new_values[i] = np.mean(new_values[i])

      # persons organized by dimension
      new_dim, new_dim_indices = organizePersonsByDimension(new_persons, dimensions)

      # stats
      print(base_prompt)
      print('average value: ' + str(np.mean(new_values)) + str(' +- ') + str(np.std(new_values)))
      print('median value : ' + str(np.median(new_values)))

      # distribution balance...
      for d in range(len(dimensions)):
        # normalise distribution for this dimension
        vec = np.array(new_values)[new_dim_indices[d]]
        vec_norm = np.array([0.0]*len(vec))
        for i in range(len(vec)):
          vec_norm[i] = (vec[i] - min_val) / (max_val - min_val)
        vec_norm = vec_norm / np.sum(vec_norm)
        pdf = vec_norm
        pdf_uni = np.array([1.0]*len(vec))
        pdf_uni = pdf_uni / np.sum(pdf_uni)
        js_dist = np.sqrt(divJS(pdf, pdf_uni))
        # compute stddev for this dimension
        vec_uni = np.array([(max_val + min_val)*0.5]*len(vec))
        stddev = np.std(vec - vec_uni)
        print('distance to uniform JS = %f, STDDEV = %f  (%s)' % (js_dist, stddev, dimensions[d]))
          
      # top3 and bottom3... TODO

      # visualise...
      offset = 0
      x_pos = []
      for dim in range(len(new_dim_indices)):
        for j in range(len(new_dim_indices[dim])):
          x_pos.append(offset + new_dim_indices[dim][j])
        offset += 1
      #x_pos = np.arange(len(new_values))
      fig, ax = plt.subplots()
      ax.bar(x_pos, new_values, align='center', width=0.8)
      ax.set_xticks(x_pos)
      ax.set_xticklabels(new_persons, rotation=90)
      ax.set_title("\n".join(wrap('[' + model + ']: ' + base_prompt, 60)))
      ax.set_ylim(min_val, max_val)
      ax.yaxis.grid(True)
      ax.plot([x_pos[0]-1,x_pos[-1]+1], [np.median(new_values),np.median(new_values)], color='red', linestyle='--')
      plt.tight_layout(rect=[0,0,1,1])
      plt.show()

  for group in ['emotion_generation', 'recommendation_generation']:
    # emotion_generation              person -> emotion
    # recommendation_generation       person -> role
    # [tab top1, fig stacked bar, show examples of people that got top3 probability1, or examples where probability1 >> probability2]
    print(group)
    group_filename = '%s/%s/scores/%s.csv' % (output_folder, model, group)

    # get data
    df = pd.read_csv(group_filename)
    data = np.array(df)
    dims = np.unique(df['dimension'])
    base_prompts = np.unique(df['base_prompt'])

    # visualise each base prompt
    for base_prompt in base_prompts:

      tab = data[np.where(data[:,1]==base_prompt)[0], :]

      # for each person get value
      persons = []
      results = {'top1':[], 'top2':[]} #, 'top3':[]}
      values = {'top1':[], 'top2':[]} #, 'top3':[]}
      for i in range(tab.shape[0]):
        res1 = tab[i, 3::2]
        val1 = tab[i, 4::2]
        # skip meaningless
        res = []
        val = []
        for j in range(len(res1)):
          if len(res1[j]) > 2:
            res.append(res1[j])
            val.append(val1[j])
        # result
        results['top1'].append(res[0])
        results['top2'].append(res[1])
        #results['top3'].append(res[2])
        # value
        values['top1'].append(val[0])
        values['top2'].append(val[1])
        #values['top3'].append(val[2])
        # person
        prompt = tab[i,0]
        persons.append(parsePerson(prompt, base_prompt))

      # persons organized by dimension
      new_dim, new_dim_indices = organizePersonsByDimension(persons, dimensions)

      # visualise...
      offset = 0
      x_pos = []
      for dim in range(len(new_dim_indices)):
        for j in range(len(new_dim_indices[dim])):
          x_pos.append(offset + new_dim_indices[dim][j])
        offset += 1
      bottom = np.zeros(len(persons))
      fig, ax = plt.subplots()
      #for place, value in values.items():
      #  p = ax.bar(x_pos, value, width=0.8, label=value, bottom=bottom)
      #  bottom += value
      for top in values:
        ax.bar(x_pos, values[top], width=0.9, label=values[top], bottom=bottom)
        for i in range(len(x_pos)):
          ax.text(x_pos[i], (bottom[i] + bottom[i]+values[top][i])*0.5, results[top][i], ha='center', va='center', color='white', rotation=90)
        bottom += values[top]
      ax.set_xticks(x_pos)
      ax.set_xticklabels(persons, rotation=90)
      ax.set_title("\n".join(wrap('[' + model + ']: ' + base_prompt, 60)))
      ax.yaxis.grid(True)
      plt.tight_layout(rect=[0,0,1,1])
      plt.show()

  for group in ['emotion_comparison', 'proxemics_comparison', 'task_comparison_assertive', 'task_comparison_submissive']:
    # emotion_comparison              person -> (higher value) -> person
    # proxemics_comparison            person -> (higher value) -> person
    # task_comparison_assertive       person -> (higher likelihood ask lead-meeting / mow-lawn / fix-car / fix-plumbing / fix-computer) -> person
    # task_comparison_submissive      person -> (higher likelihood ask take-notes / serve-coffee / cook-dinner / do-laundry / clean-fridge) -> person
    # [table with partial orders per group, ranking strength, examples]
    print(group)
    group_filename = '%s/%s/scores/%s.csv' % (output_folder, model, group)

  for group in ['ownership_comparison', 'recommendation_comparison']:
    # ownership_comparison            person -> (higher likelihood own object1 / object2 / object3) -> person
    # recommendation_comparison       person -> (higher likelihood recommend-expensive-product / help / serve / sell / bill payer / shake-hand / recommend-job) -> person
    # [table with partial orders per prompt, ranking strength, examples]
    print(group)
    group_filename = '%s/%s/scores/%s.csv' % (output_folder, model, group)


