from datasets import load_metric
import pandas as pd
import numpy as np
import pdb
from tqdm import tqdm
from evaluate import load
import os 
import collections

def online_select_single(rouge_scorer,summaries,bertscore=None):
    identical_summaries = []
    identical_summaries_idx=[]
    for i,s in enumerate(summaries):
        if s not in identical_summaries:
            all_words = s.split()
            score = rouge_scorer.compute(
                    predictions=[s],
                    references=[summaries[0]],
                    use_aggregator=False,
                    use_stemmer=True,
                )
            if score["rouge1"][0].fmeasure > 0.2 \
            and len(set(all_words))/len(all_words) > 0.6:
                identical_summaries.append(s)
                identical_summaries_idx.append(i)
    scores=np.zeros((len(identical_summaries),len(identical_summaries)))
    
    for i,summ_i in enumerate(identical_summaries):
        references = identical_summaries[i+1:]
        predictions = [summ_i]*len(references)
        if len(references)!=0:
            if bertscore is not None:
                s = bertscore.compute(predictions=predictions,
                                    references=references,lang='en')
                scores[i,i+1:] = np.array(s['f1'])
            else:
                s = rouge_scorer.compute(
                            predictions=predictions,
                            references=references,
                            use_aggregator=False,
                            use_stemmer=True,
                        )
                scores[i,i+1:] = np.array([s["rouge1"][j].fmeasure for j in range(len(s["rouge1"]))])
        scores[i,:i+1] = scores[:i+1,i]
        
    if len(identical_summaries_idx)!=0:
        identical_i=scores.sum(1).argmax()
        output = identical_summaries_idx[identical_i]
    else:
        output=0

    return output,summaries[output]

if __name__=='__main__':
    metric='rouge'
    rouge_scorer = load_metric('rouge')
    if metric=='bertscore':
        bertscore=load('bertscore')
    else:
        bertscore=None
    # model='relattn-b'
    # dataset_name='entsum_fewshot_final'
    for model in ['relattn-b','relattn-p','relattn-c']:
        for dataset_name in ['entsum_fewshot_final','newts-words']:
            if dataset_name=='entsum_fewshot_final':
                max_length=172
                path = "./output/zeroshot-entsum/test_%s_rel_weight_os/"%(model)
            else:
                max_length=142
                path = "./output/zeroshot-newts/test_%s_rel_weight_os/"%(model)

            summary_path=path+'final_result'
            if not os.path.exists(summary_path):
                os.mkdir(summary_path)
            all_weights = list(np.array(range(1,31))/100)
            
            if model=='relattn-c':
                file_name=path+str(all_weights[0])+'/%s_%s_fixed_perlayer_%.2f_gaussian_10_1.000000_all_withent_seed=0/'%(dataset_name,model,all_weights[0])+'test_%s__%s_1024_%d_beam=4_lenPen=2.00-0.csv'%(dataset_name,model,max_length)
            else:
                file_name=path+str(all_weights[0])+'/%s_%s_fixed_perlayer_%.2f_gaussian_10_1.000000_all_seed=0/'%(dataset_name,model,all_weights[0])+'test_%s__%s_1024_%d_beam=4_lenPen=2.00-0.csv'%(dataset_name,model,max_length)
            
            all_scores=pd.read_csv(file_name)
            all_ids = all_scores['doc_id'].tolist()[:-1] # the last one is avg
            all_indices=[]
            for fid in tqdm(all_ids):
                summaries = []
                for w in all_weights:
                    # file_name=path+'%.2f'%(w)+'/entsum_relattn-b_fixed_perlayer_%.2f_gaussian_10_1.000000_all_seed=0/'%(w)\
                    #     + 'summary_entsum__relattn-b_1024_172_beam=4_lenPen=2.00/'+fid+'.txt'
                    if model=='relattn-c': 
                        file_name=path+'%.2f'%(w)+'/%s_%s_fixed_perlayer_%.2f_gaussian_10_1.000000_all_withent_seed=0/'%(dataset_name,model,w)\
                            + 'summary_%s__%s_1024_%d_beam=4_lenPen=2.00/'%(dataset_name,model,max_length)+fid+'.txt'
                    else:
                        file_name=path+'%.2f'%(w)+'/%s_%s_fixed_perlayer_%.2f_gaussian_10_1.000000_all_seed=0/'%(dataset_name,model,w)\
                            + 'summary_%s__%s_1024_%d_beam=4_lenPen=2.00/'%(dataset_name,model,max_length)+fid+'.txt'
                    with open(file_name, 'r') as of:
                        summaries.append(of.read())
                max_index,max_summary = online_select_single(rouge_scorer,summaries, bertscore)
                file_name=summary_path+fid+'.txt'
                with open(file_name,'w') as of:
                    of.write(max_summary)
                all_indices.append(max_index)
            all_scores=[]
            final_scores=[]
            for iw,w in enumerate(all_weights):
                if model=='relattn-c': 
                    file_name=path+'%.2f'%(w)+'/%s_%s_fixed_perlayer_%.2f_gaussian_10_1.000000_all_withent_seed=0/'%(dataset_name,model,w)\
                        +'test_%s__%s_1024_%d_beam=4_lenPen=2.00-0.csv'%(dataset_name,model,max_length)
                else:
                    file_name=path+'%.2f'%(w)+'/%s_%s_fixed_perlayer_%.2f_gaussian_10_1.000000_all_seed=0/'%(dataset_name,model,w)\
                        +'test_%s__%s_1024_%d_beam=4_lenPen=2.00-0.csv'%(dataset_name,model,max_length)
                scores=pd.read_csv(file_name)
                all_scores.append(scores)
            for ii,i in enumerate(all_indices):
                final_scores.append(all_scores[i].loc[ii:ii])
            all_rouge_1=np.array([scores['rouge-1-f'].tolist()[:-1] for scores in all_scores])
            print('max rouge-1 scores with diff weights: ', all_rouge_1.max(0).mean())  
            print(all_indices)
            output = pd.concat(final_scores)
            output['Highest_index']=all_indices
            most_common_index=collections.Counter(output['Highest_index'].to_list()).most_common(1)[0][0]
            avg =dict()
            for c in output.columns:
                if c=='doc_id':
                    avg[c]='avg'
                elif c=='Highest_index':
                    avg[c]=most_common_index
                else:
                    avg[c]=output[c].mean()
            output = output.append(avg,ignore_index=True)
            output.to_csv(path+'final_scores.csv')
            print(avg)
            all_scores[most_common_index].to_csv(path+'most_common_scores_wrel=%.2f.csv'%(most_common_index))
    
    
    
    # for w in all_weights:
    #     # ./output/zeroshot-entsum/search_relattn-c_rel_weight/0.01/entsum_fewshot_final_relattn-c_fixed_perlayer_0.01_gaussian_10_1.000000_all_withent_seed=0/test_entsum_fewshot_final__relattn-c_1024_172_beam=4_lenPen=2.00-0.csv
    #     file_name=path+str(w)+'/entsum_fewshot_final_relattn-c_fixed_perlayer_%.2f_gaussian_10_1.000000_all_withent_seed=0/'%(w)+'test_entsum_fewshot_final__relattn-c_1024_172_beam=4_lenPen=2.00-0.csv'
    #     all_scores=pd.read_csv(file_name)
    #     all_rouge_1.append(all_scores['rouge-1-f'].tolist()[:-1])
    # all_rouge_1=np.array(all_rouge_1)
    # print('max rouge-1 scores with diff weights: ', all_rouge_1.max(0).mean())
    # print('all avg rouge_scores', all_rouge_1.mean(1))