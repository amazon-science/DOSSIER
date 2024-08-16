from textwrap import dedent

HEADER = dedent('''\
Given the following SQL tables, your job is to output a valid SQL query which can be used to validate a user's natural language claim.  
Your query should return a table containing the clinical record(s) which act as supporting evidence, and which may be used to prove or disprove the claim.
You should also output non-negative scalar values in <lower></lower> and <upper></upper> tags, and a stance character in the <stance></stance> tags.
The stance value should be a single character, either T (indicating true) or F (indicating false).
When the number of rows in the returned table is between the lower and upper bounds (inclusive), the claim should have veracity equal to the stance.
If the upper bound is positive infinity, you can leave the <upper></upper> value blank.
Output a SQL query only if the claim is verifiable and you are confident in the generated query; otherwise tell me you don't know. Do not hallucinate any clauses.''')

SCHEMA = dedent('''\
CREATE TABLE Admission ( t REAL, str_label TEXT );
CREATE TABLE Vital ( t REAL, Value REAL, Units TEXT, str_label TEXT );    
CREATE TABLE Lab ( t REAL, Value REAL, Units TEXT, str_label TEXT );
CREATE TABLE Input ( t REAL, Amount REAL, Units TEXT, str_label TEXT );
CREATE TABLE Global_KG (Subject_Name TEXT, Predicate TEXT, Object_Name TEXT );''')

PROBLEM_SPECS = dedent('''\
Here are some more details about the problem:
- t is given in hours.
- The patient was admitted to the hospital at t=0.
- Rows may not be sorted.
- The Input table contains medication and IV inputs.
- The Admission table has one row for each admission diagnosis, and is always measured at t=0.
- The Vital table contains vital measurements from the patient's chart, and the Lab table contains laboratory measurements.
- The Vital, Admission, Lab, and Input tables correspond to the electronic health records of a patient's ICU stay.
- The Global_KG table corresponds to triplets from a large biomedical knowledge graph. The triplets have the form (Subject_Name, Predicate, Object_Name). You should almost always use this table.
- Always specify a predicate when querying Global_KG.
- The Predicate column of Global_KG has the following possible values: {}
- Be very careful about whether an entity is a Subject or Object in Global_KG, particularly for the ISA predicate.
- Your query should always start with "SELECT *". Do not SELECT COUNT.
- Use the <thinking></thinking> XML tags to write down any intermediate steps.
- Please put your returned SQL query in <sql></sql> XML tags.
- Please put the veracity of the claim when the number of rows in the returned table is between <lower></lower> and <upper></upper> as a single character (T or F) in <stance></stance> XML tags.''')

EX1 = '''<example>
H: Claim made at t=70: pt was given a blood thinner in the past 24 hours.


A: <thinking>
- 'In the past 24 hours' means between t=46 and t=70, since the claim is made at t=70.
- To verify this claim, we need to check if any anticoagulant medications were given to the patient in the Input table between t=46 and t=70. 
- If the number of rows returned is between 1 and infinity (inclusive), the claim is true
- We can identify anticoagulant medications using the Global_KG table and looking for medications where 'anticoagulant' appears in the Object_Name column.
</thinking>
<sql>
SELECT *
FROM Input
JOIN Global_KG ON Input.str_label = Global_KG.Subject_Name
WHERE Global_KG.Predicate = 'ISA' 
    AND UPPER(Global_KG.Object_Name) like '%ANTICOAGULANT%'
    AND Input.t BETWEEN 46 AND 70
</sql>
<lower>1</lower>
<upper></upper>
<stance>T</stance>
</example>'''

EX2 = '''<example>
H: Claim made at t=100: pt did not have a systolic blood pressure measurement above 140 since t=20.


A: <thinking>
To verify this claim, we need to check if the patient had any systolic blood pressure measurements above 140 since t=20. 
We want to select all rows from Vital where:
- str_label is similar to "systolic pressure" 
- t is greater than 20
- Value is greater than 140
If this query returns any rows, then the claim is false.
</thinking>
<sql>
SELECT * 
FROM Vital
WHERE UPPER(str_label) like '%SYSTOLIC PRESSURE%' AND t >= 20 AND Value > 140
</sql>
<lower>1</lower>
<upper></upper>
<stance>F</stance>
</example>'''

EX3 = '''<example>
H: Claim made at t=50: pt was adminstered warfarin at most three times

A: <thinking>
- The claim is that the patient was administered warfarin at most three times
- To verify this claim, we need to find the instances where warfarin or any subtype of warfarin was administered to the patient in the Input table.
- We can use the Global_KG table to find subtypes of warfarin by looking for rows where 'warfarin' appears in the Object_Name column and the predicate is ISA.
- If the number of times warfarin was given is between 0 and 3, the claim is true.
</thinking>
<sql>
SELECT *
FROM Input 
JOIN Global_KG ON Input.str_label = Global_KG.Subject_Name
WHERE Global_KG.Predicate = 'ISA' AND UPPER(Global_KG.Object_Name) like '%WARFARIN%'
</sql>
<lower>0</lower>
<upper>3</upper>
<stance>T</stance>
</example>'''


EX4 = '''<example>
H: Claim made at t=130.5: patient was not given Lactated Ringers exactly 4 times 

A: <thinking>
- To check if Lactated Ringer's was given, we can search for it in the Input table
- If the number of rows returned is equal to 4, the claim is false. Otherwise, it is true.
</thinking>
<sql>
SELECT * 
FROM Input
JOIN Global_KG ON Input.str_label = Global_KG.Subject_Name  
WHERE UPPER(Global_KG.Object_Name) like '%LACTATED RINGERS%' AND Global_KG.Predicate = 'ISA'
</sql>
<lower>4</lower>
<upper>4</upper>
<stance>F</stance>
</example>'''

EXAMPLES = dedent(f'''\
Here are some examples:
{EX1}

{EX2}

{EX3}

{EX4}''')

MINI_EXAMPLES = dedent(f'''\
Here are some examples:
{EX1}

{EX2}''')