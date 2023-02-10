# InterpretME Example French Royalty Constraints

The directory `spouse` contains the translations of 10 logical rules into SHACL constraints.
Each rule is represented as a separate SHACL shape schema and, hence, has its own directory.
The rules are explained below:

 1. The relationship `spouse` is irreflexive, i.e., one cannot be married to himself/herself.<br>$spouse(x, y)\ \land\ x \neq y \Rightarrow hasSpouse(x)$ 
 2. In order to have a spouse, one needs to have a child which also has another parent.<br>$child(x, y)\ \land\ parent(y, z)\ \land\ x \neq z \Rightarrow hasSpouse(x)$
 3. If two people have a child, they are likely to have a spouse.<br>$child(x, y)\ \land\ child(z, y)\ \land\ x \neq z \Rightarrow hasSpouse(x)$
 4. If a person has two parents, they are likely to be married.<br>$parent(y, x)\ \land\ parent(y, z)\ \land\ x \neq z \Rightarrow hasSpouse(x)$
 5. If a person has a child whose predecessor is another person, the parent probably has a spouse.<br>$child(x, y)\ \land\ predecessor(y, z)\ \land\ x \neq z \Rightarrow hasSpouse(x)$
 6. If a person has a child who is the successor of another person, the parent probably has a spouse.<br>$child(x, y)\ \land\ successor(z, y)\ \land\ x \neq z \Rightarrow hasSpouse(x)$
 7. If a person has a father and a mother, the father has a spouse.<br>$father(y, x)\ \land\ mother(y, z)\ \land\ x \neq z \Rightarrow hasSpouse(x)$
 8. If a person has a father and a mother, the mother has a spouse.<br>$mother(y, x)\ \land\ father(y, z)\ \land\ x \neq z \Rightarrow hasSpouse(x)$
 9. If a person has a child and the child's father is a different person, then the mother has a spouse.<br>$child(x, y)\ \land\ father(y, z)\ \land\ x \neq z \Rightarrow hasSpouse(x)$
10. If a person has a child and the child's mother is a different person, then the father has a spouse.<br>$child(x, y)\ \land\ mother(y, z)\ \land\ x \neq z \Rightarrow hasSpouse(x)$