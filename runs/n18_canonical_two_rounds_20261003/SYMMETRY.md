# Exhaustive two-round reduction for 18 players

Four rounds use sizes (3,3,4,4,4). Unique partners imply every intersection
between groups in different rounds contains at most one player. Consequently
the first-two-round intersection matrix is binary, with row and column sums
(3,3,4,4,4). Exhaustively choosing each row's zero columns and checking column
sums gives 258 matrices. Simultaneous permutations of the two triple slots and
three four-player slots, together with matrix transposition, partition these
into 26 classes. Exactly one class has diagonal trace zero and one trace one.
The enumeration and representatives are saved in enumeration.json.

For minimum individual spread at least three, each player either uses four
different slots or repeats exactly one slot twice. Thus the sum of the six
round-pair diagonal traces is exactly 72 minus total spread. A total at least
66 implies this sum is at most six: some round pair has trace at most one.
This also covers any possible minimum-four schedule.

Put that pair first by round permutation. Choose the matrix representative
using a common permutation of equal-size slots and, if required, swapping the
two rounds. Each occupied intersection cell contains exactly one player.
Globally relabel that player to the canonical player assigned to its cell.
This restores the model's fixed first-round partition and fixes the complete
second round. All these transformations preserve unique partners, balanced
group-size participation, and every individual's number of distinct slots
(up to player renaming). They preserve the triples-before-fours rule.
There are no player-specific or round-specific restrictions in these models.

Therefore the two cases trace0 and trace1 cover every schedule that improves
on the validated fairness baseline (3,65). Both cases require minimum spread
at least three and total at least 66; rounds three and four remain free.
INFEASIBLE in both would prove (3,65) globally lexicographically optimal.
UNKNOWN in either leaves the corresponding case unresolved.

For each class a saved valid (3,65) schedule is normalized, independently
validated, and forced successfully into both the integer and retained models
before the target restrictions. Its 72 integer slot assignments provide hints.
The test limit is 1200 seconds per case, seed 73, eight workers. Source
snapshots, exported models, logs and per-case results are retained here.
