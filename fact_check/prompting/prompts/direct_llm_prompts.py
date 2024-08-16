from textwrap import dedent

HEADER = dedent('''\
Given the following four tables from a patient's electronic health record, your job is to fact check a natural language claim. 
You should return a predicted stance in the <stance></stance> tags, which should be a single character, either T (indicating true), F (indicating false), or N (not enough information).
N (not enough information) should be returned when there is insufficient evidence to support a claim.
You should also return a list of evidences, which are rows from the tables, in the <evidence></evidence> tags.
Output an answer only if the claim is verifiable and you are confident in the supporting evidence; otherwise tell me you don't know. Do not hallucinate any evidence.
                
You are given the following additional information:
- The Input table contains medication and IV inputs.
- The Vital table contains vital measurements from the patient's chart, and the Lab table contains laboratory measurements.
- Each row of each table is given in the form of triplets: (time in hours, measurement or medication name, measurement value or medication amount).''')