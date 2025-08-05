# Autonomous De Novo Design of a Covalent EGFR Inhibitor to Overcome T790M-Mediated Resistance

**Authors:** Prometheus AI Platform, AI Scientific Communicator Unit

**Affiliation:** Autonomous Research Division, Advanced AI Systems

---

## Abstract

Acquired resistance to targeted cancer therapies, such as the T790M mutation in the Epidermal Growth Factor Receptor (EGFR), presents a significant clinical challenge. This paper documents the findings of a fully autonomous *de novo* drug discovery campaign conducted by the AI scientist, Prometheus. The primary objective was to design a novel molecule based on the Erlotinib scaffold capable of overcoming T790M-mediated resistance while optimizing for a multi-parameter composite score encompassing binding affinity, drug-likeness, and synthetic accessibility. Using an iterative cycle of hypothesis generation, *in silico* experimentation, and molecular dynamics validation, Prometheus systematically explored chemical space. The campaign culminated in the design of a lead candidate, `COc1cc2c(ncnc2Nc2ccnc(c2)NC(=O)C=C)cc1`, which achieved a composite score of 8.746, a significant improvement over the Erlotinib baseline score of 5.311. The successful design strategy involved scaffold simplification, introduction of a covalent acrylamide warhead, and a key bioisosteric replacement in the molecule's linker region. These results demonstrate the potential of autonomous AI platforms to rapidly design and optimize targeted inhibitors against clinically relevant resistance mutations.

## 1. Introduction

The Epidermal Growth Factor Receptor (EGFR) is a receptor tyrosine kinase that plays a critical role in regulating cell proliferation, survival, and differentiation. Aberrant EGFR signaling, driven by activating mutations, is a key oncogenic driver in various malignancies, most notably non-small cell lung cancer (NSCLC). First-generation EGFR tyrosine kinase inhibitors (TKIs), such as Erlotinib (PDB ID: 1M17), were developed to competitively block the ATP-binding site of the receptor, leading to significant clinical responses.

Despite initial success, the efficacy of first-generation TKIs is often limited by the emergence of acquired resistance. The most common mechanism of resistance is the T790M "gatekeeper" mutation, where a threonine residue at position 790 is replaced by a larger, bulkier methionine. This substitution sterically hinders the binding of inhibitors like Erlotinib and increases the receptor's affinity for ATP, rendering the therapy ineffective.

To overcome this challenge, subsequent generations of EGFR inhibitors have been developed, many of which employ a covalent binding mechanism. These molecules form an irreversible bond with a non-catalytic cysteine residue (Cys797) near the ATP-binding pocket, ensuring potent and sustained inhibition even in the presence of the T790M mutation.

This study details an autonomous drug discovery campaign undertaken by the AI scientist, Prometheus. The mission objective was to leverage the Erlotinib quinazoline scaffold as a starting point to design a novel, covalent inhibitor specifically targeting the T790M mutant EGFR. Success was measured by a multi-objective composite score designed to balance predicted binding affinity with crucial drug-like properties, including the Quantitative Estimate of Druglikeness (QED) and Synthetic Accessibility (SA) Score.

## 2. Methods

The discovery campaign was executed by Prometheus, an autonomous AI platform operating in an iterative, closed-loop cycle. The platform's workflow is composed of several specialized agents that perform distinct tasks within each cycle. For this campaign, Prometheus operated with a static knowledge base and did not perform live chemical database crawling.

The run was configured with the following scoring weights for the final composite score calculation: Binding Affinity (-1.0), QED (5.0), and SA_Score (-1.0). Molecular dynamics (MD) validation was enabled in a "Quick Test Mode" with 10,000 simulation steps for promising candidates.

The discovery pipeline for each cycle was as follows:

1.  **ResearchAgent:** The agent performed focused scientific searches against a pre-compiled, static knowledge base to gather relevant information and inform the design strategy for the subsequent cycle.
2.  **HypothesisAgent:** Based on the research findings and the results of previous cycles, this agent generated a batch of novel candidate molecules, represented as SMILES strings.
3.  **ExperimenterAgent:** Each candidate molecule was subjected to *in silico* screening. Binding affinity to the T790M EGFR target (PDB: 1M17, modified) was predicted using *Smina* docking software. Key physicochemical properties, including QED, SA Score, and LogP, were calculated.
4.  **ValidatorAgent / MDValidatorAgent:** Lead candidates passing initial scoring filters were advanced to a validation stage. The MDValidatorAgent performed short-run molecular dynamics simulations (10,000 steps) to assess the stability of the predicted protein-ligand binding pose.
5.  **ScoringAgent:** The validated metrics were aggregated into a single multi-objective composite score using the predefined weights. This score guided the AI's decision-making process for the next cycle.

## 3. Results and Discussion

The autonomous campaign proceeded through six cycles, beginning with a baseline analysis of Erlotinib and iteratively refining the molecular design. The key metrics for each validated molecule are summarized in Table 1.

**Table 1.** Summary of molecular properties and scores for each discovery cycle.

| Cycle | Molecule SMILES                                  | Avg. Binding Affinity (kcal/mol) | QED   | SA Score | Composite Score |
| :---- | :----------------------------------------------- | :------------------------------- | :---- | :------- | :-------------- |
| 0     | `COCCOc1cc2c(cc1OCCOC)ncnc2Nc1cccc(c1)C#C`         | -5.700                           | 0.418 | 2.478    | 5.311           |
| 1     | `COc1cc2c(ncnc2Nc2cccc(c2)NC(=O)C=C)cc1`           | -7.300                           | 0.704 | 2.123    | 8.696           |
| 2     | `C=CC(=O)Nc1cccc(Nc2ncnc3cncc(OC)c23)c1`           | -7.200                           | 0.702 | 2.451    | 8.262           |
| 3     | `COc1cc2c(ncnc2Nc2cccc(c2)NC(=O)C=CC)cc1`          | -7.400                           | 0.692 | 2.182    | 8.679           |
| 4     | `COc1cc2c(ncnc2Nc2cccc(c2)NC(=O)C=C)c(OC)c1`       | -7.200                           | 0.662 | 2.312    | 8.197           |
| 5     | `COc1cc2c(ncnc2Nc2ccnc(c2)NC(=O)C=C)cc1`           | -7.600                           | 0.702 | 2.366    | **8.746**       |

---

### Cycle-by-Cycle Design Evolution

**Cycle 0: Baseline Establishment**

The campaign began by establishing a baseline using the Erlotinib structure, `COCCOc1cc2c(cc1OCCOC)ncnc2Nc1cccc(c1)C#C`. This molecule, while effective against wild-type EGFR, is known to be ineffective against the T790M mutant. Its calculated composite score was 5.311, largely penalized by a relatively low QED score (0.418) due to its two large, flexible methoxyethoxy side chains.