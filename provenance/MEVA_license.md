# MEVA license and attribution (Topic 09)

Access date for all sources below: **2026-09-25**.

## Verdict

MEVA (video on `s3://mevadata-public-01`, annotations and metadata in the Kitware
meva-data-repo) is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
Research use, derived artefacts (features, temporal profiles, subsets) and publication are
permitted, including commercial use, provided attribution is given: name the creators
(Kitware Inc. and IARPA), link the license, link the material, and indicate modifications.
No non-commercial or share-alike restriction applies. CC BY 4.0 does not license publicity/privacy
rights (Section 2(b)(1)), so figures showing identifiable people should be used with care.

## Sources

| Source | URL |
|---|---|
| MEVA homepage, "ACCESSING AND USING MEVA / License" | https://mevadata.org/ |
| MEVA data license text linked from the homepage | https://mevadata.org/resources/MEVA-data-license.txt |
| Kitware MEVA data repo (annotations, metadata) | https://gitlab.kitware.com/meva/meva-data-repo |
| Repo LICENSE file (commit 421841a75577b697c314e952e585aecbb1b99e17, 2021-12-13) | https://gitlab.kitware.com/meva/meva-data-repo/-/blob/master/LICENSE |
| CC BY 4.0 legal code | https://creativecommons.org/licenses/by/4.0/ |

The license text at mevadata.org/resources/MEVA-data-license.txt and the repo `LICENSE` file are
identical in content (checked with `diff` after stripping CR line endings).

## Verbatim quotes

### mevadata.org (homepage, section "ACCESSING AND USING MEVA")

> License
>
> All MEVA data is available for use under a CC BY-4.0 license; the general MEVA data license is available here.

("CC BY-4.0" links to https://creativecommons.org/licenses/by/4.0/; "here" links to
https://mevadata.org/resources/MEVA-data-license.txt.)

### License file header (mevadata.org/resources/MEVA-data-license.txt = repo LICENSE, lines 1-4)

> "Multiview Extended Video with Activities" (MEVA) dataset by Kitware Inc. and the Intelligence Advanced Research Projects Activity (IARPA).
>
> "Multiview Extended Video with Activities" (MEVA) dataset by Kitware Inc. and the Intelligence Advanced Research Projects Activity (IARPA) is licensed under a Creative
> Commons Attribution 4.0 International License (http://creativecommons.org/licenses/by/4.0.)

### Attribution requirement (same file, CC BY 4.0 Section 3(a), verbatim)

```
Section 3 -- License Conditions.

Your exercise of the Licensed Rights is expressly made subject to the
following conditions.

  a. Attribution.

       1. If You Share the Licensed Material (including in modified
          form), You must:

            a. retain the following if it is supplied by the Licensor
               with the Licensed Material:

                 i. identification of the creator(s) of the Licensed
                    Material and any others designated to receive
                    attribution, in any reasonable manner requested by
                    the Licensor (including by pseudonym if
                    designated);

                ii. a copyright notice;

               iii. a notice that refers to this Public License;

                iv. a notice that refers to the disclaimer of
                    warranties;

                 v. a URI or hyperlink to the Licensed Material to the
                    extent reasonably practicable;

            b. indicate if You modified the Licensed Material and
               retain an indication of any previous modifications; and

            c. indicate the Licensed Material is licensed under this
               Public License, and include the text of, or the URI or
               hyperlink to, this Public License.

       2. You may satisfy the conditions in Section 3(a)(1) in any
          reasonable manner based on the medium, means, and context in
          which You Share the Licensed Material. For example, it may be
          reasonable to satisfy the conditions by providing a URI or
          hyperlink to a resource that includes the required
          information.
```

### Repo README (annotation contributions)

From `annotation/DIVA-phase-2/MEVA/README.md`:

> **PLEASE NOTE** that any corrections or new annotations will be subject to the same [license](../../../LICENSE) as this overall annotation repository.

### Requested citation (mevadata.org, "Citing MEVA")

> The dataset is described in our WACV 2021 paper. The bibtex citation is:

```
@InProceedings{Corona_2021_WACV,
    author    = {Corona, Kellie and Osterdahl, Katie and Collins, Roderic and Hoogs, Anthony},
    title     = {MEVA: A Large-Scale Multiview, Multimodal Video Dataset for Activity Detection},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {1060-1068}
}
```

## Attribution line to use in Topic 09 outputs

> Contains video and annotations from the "Multiview Extended Video with Activities" (MEVA) dataset by
> Kitware Inc. and the Intelligence Advanced Research Projects Activity (IARPA), https://mevadata.org/,
> licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Modifications: a 20 h subset
> of drops-123-r13 was selected (docs/MEVA_selection.md), decoded, and resized/converted to grayscale
> for analysis. Cite Corona et al., WACV 2021.
