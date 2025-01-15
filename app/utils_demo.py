import os

# # util function that also neo4j demo databases init for prod
# def init_neo4j_databases():
#     databases = os.getenv("NEO4J_DATABASE")

#     if databases == None:
#         raise Exception("NEO4J_DATABASE env variable not set.")

#     databases = databases.split(",")

#     print(">>>>", databases)

#     database_list = []

#     if len(databases) == 1:
#         #
#         database_list.append({
#             "uri": os.getenv("NEO4J_DATABASE_URI"),
#             "database": os.getenv("NEO4J_DATABASE"),
#             "username": os.getenv("NEO4J_USERNAME"),
#             "password": os.getenv("NEO4J_PASSWORD"),
#         })
