### TESTBOOK

## Testbook for multimodal RAG (mmrag.py script)


# QUERY:

Given images, tables, policies, infrastructures, and other information about Contoso corporation, generates 10 tricky and independent questions that can be made about these topics. Use also precise information queries about numbers and percentages in tables or images, or anything else. Try also comparing tables and images content in order to create more complex queries. You can use all retrived sources to create connections, do not focus just on the first or last ones.


# QUESTIONS:

1. How does Contoso's global network infrastructure depicted in the images support the high availability and performance requirements for the SaaS applications like Office 365, as described in the network analysis and performance tables? [doc_49, doc_13]

2. Considering the data classification levels and the encryption policies outlined in the security and data classification tables, what percentage of high-value data (Level 3) is likely to be stored in Azure IaaS or PaaS environments, and what security measures are mandated for such data? [doc_28, doc_29]

3. Based on the organizational structure and office distribution images, what is the estimated number of employees in the Paris headquarters, and how does this number relate to the 15,000 employees mentioned in the same image? What implications does this have for network bandwidth planning? [doc_51]

4. How do the policies on workplace safety and violence prevention integrate with the remote and mobile workforce depicted in the organizational and infrastructure images, especially considering the policies on device management and remote access? [doc_37, doc_39, doc_50]

5. Given the network analysis and the detailed descriptions of ExpressRoute and Traffic Manager configurations, what is the expected impact on latency and throughput for the regional hubs compared to the Paris headquarters? How does this support the performance requirements for Azure PaaS applications? [doc_16, doc_21, doc_14]

6. How does the data retention policy of 6 months for low business value data compare with the 7-year data retention requirement for high-value data, and what encryption or access controls are specified for these different data levels? [doc_30, doc_27]

7. Considering the organizational roles and the emphasis on security policies, what specific responsibilities might the Chief Technology Officer and the Chief Security Officer have in implementing the multi-factor authentication and encryption policies across the global network? [doc_48, doc_24]

8. How does the network's regional hub connectivity, as shown in the images, facilitate the deployment of Azure virtual networks with non-overlapping address spaces, and what planning steps are necessary to support the short-term increase in servers for quarter-end processing? [doc_15, doc_22]

9. Based on the policies for data security and the organizational structure, what procedures might be in place for handling a security breach involving customer PII data stored in Azure, considering the encryption and access control policies? [doc_24, doc_29]

10. How do the policies on whistleblower reporting and workplace safety influence the design of Contoso’s internal security and incident response protocols, especially in the context of the global and distributed nature of its infrastructure? [doc_45, doc_41]