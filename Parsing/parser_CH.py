# ECE467 Project 2: Parser
# By Colin Hwang

import string

# cnf grammar class to store information within elements
# follow cnf rules A -> B C or A -> w
# will be used later to backtrack to print out parses
class CNFgrammar:
    def __init__(self, left, right1=None, right2=None):
        self.left = left
        self.right1 = right1
        self.right2 = right2

# reads cnf grammar file and creates grammar rules
def dictionary():
    # get cnf grammar file from user
    cnfFile = input("Enter the name of the cnf file: ")
    grammarFile = open(cnfFile, 'r')
    # go through grammar file and add rules
    grammarRules = {}
    for rule in grammarFile.read().splitlines():
        if rule != "":
            # left and right sides of rules separated by -->
            left, right = rule.split(" --> ")
            if right in grammarRules:
                grammarRules[right].append(left)
            else:
                grammarRules[right] = [left]
    return grammarRules

# CKY algorithm using dynamic programming
def parse_sentence(sentence, grammarRules):
    # split each word in the sentence into an array
    words = sentence.split()
    # two deminsional len(token) + 1 x len(token) + 1 array to act as a table for DP
    dp = [[[] for x in range(len(words) + 1)] for y in range(len(words) + 1)]
    # using the pseudocode from the textbook (slide 26 from parsing PowerPoint)
    for j in range(1, len(words) + 1):
        word = words[j - 1]
        # get non-terminals that get derive this word
        for left in grammarRules.get(word, []):
            dp[j - 1][j].append(CNFgrammar(left, CNFgrammar(word)))
        # get parses for substring [i,j] that splits k
        for i in range(j - 2, -1, -1):
            for k in range(i + 1, j):
                for nt1 in dp[i][k]:
                    for nt2 in dp[k][j]:
                        for left in grammarRules.get(" ".join([nt1.left, nt2.left]), []):
                            dp[i][j].append(CNFgrammar(left, nt1, nt2))
    return dp

# preorder depth first search to print parse tree with proper indenting
# print only bracketed notation if display tree is false
def preorderDFS(parse, displayTree, indent):
    tab = '\t'
    if displayTree:
        nextLine = ("\n" + (tab * indent))
    else:
        nextLine = " "
    if indent > 0:
        print(nextLine, end="")
    if parse.right2 is None:
        print("[" + parse.left + " " + parse.right1.left + "]", end="")
    else:
        print("[" + parse.left, end="")
        preorderDFS(parse.right1, displayTree, indent + 1)
        preorderDFS(parse.right2, displayTree, indent + 1)
        if displayTree:
            print(nextLine, end="")
        print("]", end="")

if __name__ == "__main__":
    grammarRules = dictionary()
    while True:
        sentence = str(input("Enter a sentence to be parsed or type quit to exit the program: "))
        # if the user types "quit", exit program
        if sentence == "quit":
            break
        dp = parse_sentence(sentence, grammarRules)
        # check for valid parses
        parseTrees = len(list(filter(lambda rule: rule.left == "S", dp[0][-1])))
        if parseTrees == 0:
            print("NO VALID PARSES")

        else:
            print("VALID SENTENCE\n")
            # retrieve parse tree in bracket notation if the user wants it
            if input("Do you want textual parse trees to be displayed (y/n)?: ").lower() == "y":
                displayTree = True
            else:
                displayTree= False
            for num, element in enumerate(list(filter(lambda rule: rule.left == "S", dp[0][-1])), 1):
                print(f"Valid parse #{num}:")
                preorderDFS(element, False, 0)
                print("\n")
                if displayTree:
                    preorderDFS(element, True, 0)
                    print("\n")
            print(f"Number of valid parses: {parseTrees}")