import os
import math
import json
import random
import argparse
from collections import Counter
from copy import deepcopy
from distutils.util import strtobool


def getClass(dic, y):
    for a in dic:
        if y in dic[a]:
            return a

# Reverse the (key, value) in dic. Dic should be a mapping between (class, list {names})
def reverseDict(dic):
    revD = {}
    for a in dic:
        for y in dic[a]:
            revD[y] = a
    return revD


# This function get the classication result for each input key like "class1.txt"
def getResult(dicBProb, key):
    currProb = 0
    ans = None
    for y in dicBProb[key]:
        if dicBProb[key][y] > currProb:
            ans = y
            currProb = dicBProb[key][y]
    return ans


def classProb(dicA, dicB):
    dicBProb = {}
    revA = reverseDict(dicA)
        
    for x in dicB:
        temp = {}
        for y in dicB[x]:
            if not revA[y] in temp:
                temp[revA[y]] = 0
            temp[revA[y]] += 1

        # get probability
        total = len(dicB[x])
        for y in temp:
            temp[y] /= total
        dicBProb[x] = temp

    return dicBProb


# Builds a two-way probablity matrix (as a dict-pair)
def probTable(probA, probB):
    newA = {}
    newB = {}
    
    for classA in probA:
        newA[classA] = {}
        for matches in probA[classA]:
            if classA in probB[matches]:
                pA = probA[classA][matches]
                pB = probB[matches][classA]
                p = pA * pB
                
                newA[classA][matches] = p
                
                if not matches in newB:
                    newB[matches] = {}
                newB[matches][classA] = p
    return newA, newB


# This function return the crossEntropy given a relation mapping and a probability
def crossEntropy(relation, prob):
    ans = 0
    failed = False
    for subclass in prob:
        label = relation[subclass]
        if label in prob[subclass]:
            ans -= math.log(prob[subclass][label])
        else:
            failed = True
    if failed:
        return None
    
    num_clusters = len(relation.keys())
    scale = num_clusters * math.log(num_clusters)
    
    return ans / scale


def nextPermutation(nums):
    n = len(nums)
    i = n - 1
    while i > 0 and nums[i] <= nums[i-1]:
        i -= 1
    if i == 0:
        return None
    
    ind = temp = None
    for j in range(i, n):
        if temp is None or nums[i-1] < nums[j] <= temp:
            temp = nums[j]
            ind = j
    nums[i-1], nums[ind] = nums[ind], nums[i-1]
    nums[i:] = sorted(nums[i:])
    return nums


def main():
    # read the directory(relative path) and find all txt file and order them
    parser = argparse.ArgumentParser()
    
    if "SM_CHANNEL_TRUTH" in os.environ:
        parser.add_argument("--truths", type=str, default=os.environ["SM_CHANNEL_TRUTH"])
        parser.add_argument("--preds", type=str, default=os.environ["SM_CHANNEL_PRED"])
    else:
        parser.add_argument("--truths", type=str)
        parser.add_argument("--preds", type=str)
        
    parser.add_argument("--method", type=str, default="heuristic", help="Determines \
        which error algorithm to use: 'heuristic', 'permutation', 'both'")
    parser.add_argument("--out-dir", type=str, default="results/")
    parser.add_argument("--out-file", type=str, default="evaluation.json")
    parser.add_argument('--save-mapping', type=lambda x: bool(strtobool(x)), default=True, help='En(dis)able uploading results to S3')
    parser.add_argument("--output-s3", type=str, default="s3://calvinandpogs-ee148/lilabc/evaluation/")
    parser.add_argument('--save-s3', type=lambda x: bool(strtobool(x)), default=True, help='En(dis)able uploading results to S3')
    args = parser.parse_args()

    pathA = args.truths
    pathB = args.preds

    do_heuristic = False
    do_perm = False
    if args.method == "heuristic":
        do_heuristic = True
    if args.method == "permutation":
        do_perm = True
    if args.method == "both":
        do_heuristic = True
        do_perm = True


    A = sorted([fileName for fileName in os.listdir(pathA) if ".txt" in fileName])
    B = sorted([fileName for fileName in os.listdir(pathB) if ".txt" in fileName])

    if len(A) != len(B):
        raise ValueError("Number of truths({0}) and preds({1}) don't match!".format(len(A), len(B)))

    # A is the ground truth and B is the classify results
    # transform the data in dicB into probability in 
    dicA = {file: set(line.strip() for line in open(os.path.join(pathA, file))) for file in A}
    dicB = {file: set(line.strip() for line in open(os.path.join(pathB, file))) for file in B}
        
    dicAProb = classProb(dicB, dicA)
    dicBProb = classProb(dicA, dicB)
    BProb = deepcopy(dicBProb)
    
    
    print("---------------------------EVALUATION RESULTS-------------------------")
    
    results_dict = {}

    """
    method 1: First do a sorting using the number of points in each subclass, then 
    use the maximal probability to decide the mapping and get the first/second error
    """

    if do_heuristic:
        res = {} # a one-to-one and onto map from B --> A
        
        tableA = {}
        tableB = {}
        
        for classA, probs in dicBProb.items():
            tableA[classA] = sorted(list(probs.items()), key=lambda x: x[1], reverse=True)
        
        for classB, probs in dicAProb.items():
            tableB[classB] = sorted(list(probs.items()), key=lambda x: x[1], reverse=True)
        
        for classA in tableA:
            probsB = tableA[classA]
            
            # Find best matching
            i = 0
            classB = None
            prob = 0
            for i, (classB, prob) in enumerate(probsB):
                if i == (len(probsB) - 1):     # Last possible element, keep it
                    break
                if classA in dicAProb[classB] and tableB[classB][0][0] == classA:
                    break
                    
            res[classA] = classB
                
            # Remove classB from tableA probs
            for key in tableA:
                found = False
                i = 0
                for i, (class_B, prob) in enumerate(tableA[key]):
                    if class_B == classB:
                        found = True
                        break
                if found:
                    tableA[key].pop(i)
                
            # Remove classA from tableB probs
            for key in tableB:
                found = False
                i = 0
                for i, (class_A, prob) in enumerate(tableB[key]):
                    if class_A == classA:
                        found = True
                        break
                if found:
                    tableB[key].pop(i)
                    
            # Remove classA from dicAProb probs
            for key in dicAProb:
                if classA in dicAProb[key]:
                    dicAProb[key].pop(classA)

        end_probs = {}
        total = 0
        for k,v in res.items():
            if v is not None:
                end_probs[k] = BProb[k][v]
            else:
                end_probs[k] = 0
            total += end_probs[k]
        avg_prob = total / len(end_probs.keys())
                    
        heuristic_loss = crossEntropy(res, BProb)
        loss_string = math.inf if not heuristic_loss else heuristic_loss
        print("Heuristic cluster mapping: {}".format(res))
        print("Heuristic Probs: {}".format(end_probs))
        print("Heuristic avg prob: {}".format(avg_prob))
        print("Heuristic cross entropy: {}".format(loss_string))
        
        results_dict["heuristic"] = {}
        results_dict["heuristic"]["cross_entropy"] = heuristic_loss
        results_dict["heuristic"]["avg_prob"] = avg_prob
        results_dict["heuristic"]["probs"] = end_probs
        if args.save_mapping:
            results_dict["heuristic"]["mapping"] = res
                
    """
    method 2: Cross Entropy minimization using permuation, only work for small n
    """

    if do_perm:
        labelsB = sorted(BProb.keys())

        # main code for permutation minimization
        minCrossEntropy = None
        minMapping = None
        labelsA = sorted(dicA.keys())
        while labelsB:
            relationMap = {}
            for i, x in enumerate(labelsB):
                relationMap[x] = labelsA[i]
            new_entropy = crossEntropy(relationMap, BProb)
            if (not new_entropy is None) and (not minCrossEntropy or new_entropy <= minCrossEntropy):
                minCrossEntropy = new_entropy
                minMapping = relationMap
            labelsB = nextPermutation(labelsB)
            

        end_probs = {}
        total = 0
        for k,v in res.items():
            if v is not None:
                end_probs[k] = BProb[k][v]
            else:
                end_probs[k] = 0
            total += end_probs[k]
        avg_prob = total / len(end_probs.keys())
        
        loss_string = math.inf if not minCrossEntropy else minCrossEntropy
        print("Permutation cluster mappying: {}".format(minMapping))
        print("Permutation Probs: {}".format(end_probs))
        print("Permutation avg prob: {}".format(avg_prob))
        print("Permutation cross entropy {}".format(loss_string))
        
        results_dict["perm"] = {}
        results_dict["perm"]["cross_entropy"] = minCrossEntropy
        results_dict["perm"]["avg_prob"] = avg_prob
        results_dict["perm"]["probs"] = end_probs
        if args.save_mapping:
            results_dict["perm"]["mapping"] = minMapping
        
        
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    
    with open(os.path.join(args.out_dir, args.out_file), 'w') as f:
        json.dump(results_dict, f)
        
    if args.save_s3:
        print("Executing: aws s3 cp --recursive {} {}".format(args.out_dir, args.output_s3))
        os.system("aws s3 cp --recursive {} {} >/dev/null".format(args.out_dir, args.output_s3))


if __name__ == '__main__':
    main()