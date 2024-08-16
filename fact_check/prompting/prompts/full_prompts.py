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
CREATE TABLE Admission ( t REAL, CUI TEXT, str_label TEXT );
CREATE TABLE Vital ( t REAL, CUI TEXT, Value REAL, Units TEXT, str_label TEXT );    
CREATE TABLE Lab ( t REAL, CUI TEXT, Value REAL, Units TEXT, str_label TEXT );
CREATE TABLE Input ( t REAL, CUI TEXT, Amount REAL, Units TEXT, str_label TEXT );
CREATE TABLE Global_KG ( Subject_CUI TEXT, Predicate TEXT, Object_CUI TEXT );''')

PROBLEM_SPECS = dedent('''\
Here are some more details about the problem:
- t is given in hours.
- The patient was admitted to the hospital at t=0.
- Rows may not be sorted.
- The Input table contains medication and IV inputs.
- The Admission table has one row for each admission diagnosis, and is always measured at t=0.
- The Vital table contains vital measurements from the patient's chart, and the Lab table contains laboratory measurements.
- The Vital, Admission, Lab, and Input tables correspond to the electronic health records of a patient's ICU stay.
- The Global_KG table corresponds to triplets from a large biomedical knowledge graph. The triplets have the form (Subject_CUI, Predicate, Object_CUI). You should almost always use this table.
- Always specify a predicate when querying Global_KG.
- The Predicate column of Global_KG has the following possible values: {}
- Be very careful about whether an entity is a Subject_CUI or an Object_CUI in Global_KG, particularly for the ISA predicate.
- Match on CUI (Concept Unique Identifier) whenever possible instead of str_label.
- Due to the varying levels of granularity that CUIs may be stored in the EHR, you should always use the Global_KG with the ISA predicate to check for the presence or value of any entity. The Global_KG contains self loops with the ISA predicate.
- Your query should always start with "SELECT *". Do not SELECT COUNT.
- Use the <thinking></thinking> XML tags to write down any intermediate steps.
- Please put your returned SQL query in <sql></sql> XML tags.
- Please put the veracity of the claim when the number of rows in the returned table is between <lower></lower> and <upper></upper> as a single character (T or F) in <stance></stance> XML tags.''')

EX1 = '''<example>
H: You are given the following prior knowledge:
- Potentially relevant CUIs found in the claim: ('Anticoagulants', 'C0003280'), ('Treatment given', 'C0580351'), ('Bloods', 'C0392895'), ('Thinners', 'C0556614'), ('In the past', 'C1444637'), ('24 hours', 'C1442770')
- Out of the potentially relevant CUIs, the following appear at least once in the Subject_CUI column of Global_KG: C0003280
- Out of the potentially relevant CUIs, the following appear at least once in the Object_CUI column of Global_KG: C0003280

Claim made at t=70: pt was given a blood thinner in the past 24 hours.


A: <thinking>
- 'Blood thinner' refers to anticoagulant medications. The CUI for 'Anticoagulants' is C0003280.
- 'In the past 24 hours' means between t=70-24 and t=70, since the claim is made at t=70.
- To verify this claim, we need to check if any anticoagulant medications were given to the patient in the Input table between t=70-24 and t=70. 
- If the number of rows returned is between 1 and infinity (inclusive), the claim is true
- We can identify anticoagulant medications using the Global_KG table and looking for medications where C0003280 appears in the Object_CUI column.
</thinking>
<sql>
SELECT *
FROM Input
JOIN Global_KG ON Input.CUI = Global_KG.Subject_CUI
WHERE Global_KG.Predicate = 'ISA' 
    AND Global_KG.Object_CUI = 'C0003280'
    AND Input.t BETWEEN 70-24 AND 70
</sql>
<lower>1</lower>
<upper></upper>
<stance>T</stance>
</example>'''

MINI_EX1 = '''<example>
H: You are given the following prior knowledge:
- Potentially relevant CUIs found in the claim: ('Anticoagulants', 'C0003280'), ('Treatment given', 'C0580351'), ('Bloods', 'C0392895'), ('Thinners', 'C0556614'), ('In the past', 'C1444637'), ('24 hours', 'C1442770')
- Out of the potentially relevant CUIs, the following appear at least once in the Subject_CUI column of Global_KG: C0003280
- Out of the potentially relevant CUIs, the following appear at least once in the Object_CUI column of Global_KG: C0003280

Claim made at t=70: pt was given a blood thinner in the past 24 hours.


A: <sql>
SELECT *
FROM Input
JOIN Global_KG ON Input.CUI = Global_KG.Subject_CUI
WHERE Global_KG.Predicate = 'ISA' 
    AND Global_KG.Object_CUI = 'C0003280'
    AND Input.t BETWEEN 70-24 AND 70
</sql>
<lower>1</lower>
<upper></upper>
<stance>T</stance>
</example>'''

EX2 = '''<example>
H: You are given the following prior knowledge:
- Potentially relevant CUIs found in the claim: ('Blood pressure determination', 'C0005824'), ('Systolic blood pressure', 'C0871470')
- Out of the potentially relevant CUIs, the following appear at least once in the Subject_CUI column of Global_KG: C0005824, C0871470
- Out of the potentially relevant CUIs, the following appear at least once in the Object_CUI column of Global_KG: 

Claim made at t=100: pt did not have a systolic blood pressure measurement above 140 since t=20.


A: <thinking>
To verify this claim, we need to check if the patient had any systolic blood pressure measurements above 140 since t=20. 
The CUI for systolic blood pressure is C0871470. We need to check if there are any rows in the Vital table where the CUI is C0871470 or a subtype of C0871470.
We can use the Global_KG table to find subtypes of C0871470 by looking for rows where C0871470 appears in the Object_CUI column and the predicate is ISA.
We want to select all rows from Vital where:
- CUI is an Subject_CUI of Global_KG with Object_CUI = C0871470 and Predicate = ISA
- t is greater than 20
- Value is greater than 140
If this query returns any rows, then the claim is false.
</thinking>
<sql>
SELECT * 
FROM Vital
JOIN Global_KG ON Vital.CUI = Global_KG.Subject_CUI
WHERE Global_KG.Predicate = 'ISA' AND Global_KG.Object_CUI = 'C0871470' AND t >= 20 AND Vital.Value > 140
</sql>
<lower>1</lower>
<upper></upper>
<stance>F</stance>
</example>'''

MINI_EX2 = '''<example>
H: You are given the following prior knowledge:
- Potentially relevant CUIs found in the claim: ('Blood pressure determination', 'C0005824'), ('Systolic blood pressure', 'C0871470')
- Out of the potentially relevant CUIs, the following appear at least once in the Subject_CUI column of Global_KG: C0005824, C0871470
- Out of the potentially relevant CUIs, the following appear at least once in the Object_CUI column of Global_KG: 

Claim made at t=100: pt did not have a systolic blood pressure measurement above 140 since t=20.


A: <sql>
SELECT * 
FROM Vital
JOIN Global_KG ON Vital.CUI = Global_KG.Subject_CUI
WHERE Global_KG.Predicate = 'ISA' AND Global_KG.Object_CUI = 'C0871470' AND t >= 20 AND Vital.Value > 140
</sql>
<lower>1</lower>
<upper></upper>
<stance>F</stance>
</example>'''


EX3 = '''<example>
H: You are given the following prior knowledge:
- Potentially relevant CUIs found in the claim: ('warfarin', 'C0043031')
- Out of the potentially relevant CUIs, the following appear at least once in the Subject_CUI column of Global_KG: C0043031
- Out of the potentially relevant CUIs, the following appear at least once in the Object_CUI column of Global_KG: C0043031

Claim made at t=50: pt was adminstered warfarin at most three times

A: <thinking>
- The claim is that the patient was administered warfarin at most three times
- The CUI for warfarin is C0043031.
- To verify this claim, we need to find the instances where warfarin or any subtype of warfarin was administered to the patient in the Input table.
- We can use the Global_KG table to find subtypes of warfarin by looking for rows where C0043031 appears in the Object_CUI column and the predicate is ISA.
- If the number of times warfarin was given is between 0 and 3, the claim is true.
</thinking>
<sql>
SELECT *
FROM Input 
JOIN Global_KG ON Input.CUI = Global_KG.Subject_CUI
WHERE Global_KG.Predicate = 'ISA' AND Global_KG.Object_CUI = 'C0043031'
</sql>
<lower>0</lower>
<upper>3</upper>
<stance>T</stance>
</example>'''

EX4 = '''<example>
H: You are given the following prior knowledge:
- Potentially relevant CUIs found in the claim, along with their semantic types: ("Lactated Ringer's Solution", 'C0073385', ['Organic Chemical', 'Pharmacologic Substance']), ('4 Times', 'C4264481', ['Finding']), ('Treatment given', 'C0580351', ['Finding']), ('lactate', 'C0376261', ['Organic Chemical', 'Pharmacologic Substance'])
- Out of the potentially relevant CUIs, the following appear at least once in the Subject_CUI column of Global_KG: C0073385, C0580351, C0376261
- Out of the potentially relevant CUIs, the following appear at least once in the Object_CUI column of Global_KG: C0580351, C0376261

Claim made at t=130.5: patient was not given Lactated Ringers exactly 4 times 

A: <thinking>
- Lactated Ringer's Solution has CUI C0073385
- To check if Lactated Ringer's was given, we can search for it in the Input table
- So we will query the Input table and the Global_KG table for rows where Object_CUI = C0073385
- If the number of rows returned is equal to 4, the claim is false. Otherwise, it is true.
</thinking>
<sql>
SELECT * 
FROM Input
JOIN Global_KG ON Input.CUI = Global_KG.Subject_CUI  
WHERE Global_KG.Object_CUI = 'C0073385' AND Global_KG.Predicate = 'ISA'
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
{MINI_EX1}

{MINI_EX2}''')