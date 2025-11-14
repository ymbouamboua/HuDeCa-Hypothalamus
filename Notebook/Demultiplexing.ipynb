# Vireo and Souporcell donor match

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import vireoSNP
import os
import glob
import subprocess
import tempfile

print("vireoSNP version: %s" % vireoSNP.__version__)

bulk_vcf = "/Users/yvon.mbouamboua/Documents/projects/singlecell/hudeca/data/processed/SNV/snv-bulk/renamed_joint_common.vcf.gz"
FH.S5678_donor_vcf = "/Users/yvon.mbouamboua/Documents/projects/singlecell/hudeca/data/processed/WS_demuxafy/FH.S5678/Vireo_OUTDIR/GT_donors.vireo.vcf.gz"
FH.S5678_combined_tsv = "/Users/yvon.mbouamboua/Documents/projects/singlecell/hudeca/data/processed/WS_demuxafy/FH.S5678/combine_majoritySinglet/combined_results_w_combined_assignments.tsv"
FH.S5678_outdir = "FH.S5678_vireo_bulk_mapping_out"

FH.S124_donor_vcf = "/Users/yvon.mbouamboua/Documents/projects/singlecell/hudeca/data/processed/WS_demuxafy/FH.S124/Vireo_OUTDIR/GT_donors.vireo.vcf.gz"
FH.S124_combined_tsv = "/Users/yvon.mbouamboua/Documents/projects/singlecell/hudeca/data/processed/WS_demuxafy/FH.S124/combine_majoritySinglet/combined_results_w_combined_assignments.tsv"
FH.S124_outdir = "FH.S124_vireo_bulk_mapping_out"


def run_bulk_donor_mapping(
    bulk_vcf,
    donor_vcf,
    combined_tsv,
    outdir,
    min_concordance=0.6,
    ambiguity_delta=0.05,
    multiplex_name=None,
    border_color="black",
    cmap="viridis"
):
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from cyvcf2 import VCF

    os.makedirs(outdir, exist_ok=True)

    # --------------------------------------------
    # Helper for allele count
    # --------------------------------------------
    def gt_to_ac(gt):
        try:
            a, b = gt[0], gt[1]
        except Exception:
            return np.nan
        if a is None or b is None or a < 0 or b < 0:
            return np.nan
        return a + b

    # --------------------------------------------
    # 1) Read donor VCF
    # --------------------------------------------
    donor_vcf_obj = VCF(donor_vcf)
    donor_samples = donor_vcf_obj.samples
    donor_dict = {}
    for var in donor_vcf_obj:
        key = f"{var.CHROM}:{var.POS}:{var.REF}:{','.join(var.ALT)}"
        donor_dict[key] = np.array([gt_to_ac(gt) for gt in var.genotypes], dtype=float)
    donor_vcf_obj.close()

    # --------------------------------------------
    # 2) Read bulk VCF & intersect variants
    # --------------------------------------------
    bulk_vcf_obj = VCF(bulk_vcf)
    bulk_samples = bulk_vcf_obj.samples
    bulk_rows, matched_keys = [], []

    for var in bulk_vcf_obj:
        key = f"{var.CHROM}:{var.POS}:{var.REF}:{','.join(var.ALT)}"
        if key in donor_dict:
            acs = np.array([gt_to_ac(gt) for gt in var.genotypes], dtype=float)
            if np.all(np.isnan(acs)):
                continue
            bulk_rows.append(acs)
            matched_keys.append(key)
    bulk_vcf_obj.close()

    if len(matched_keys) == 0:
        raise ValueError("No overlapping variants found between donor and bulk VCFs.")
    if len(matched_keys) < 5:
        print(f"⚠️ Warning: Only {len(matched_keys)} overlapping variants found. Concordance may be unreliable.")

    bulk_gt_matrix = np.vstack(bulk_rows)
    donor_gt_matrix = np.vstack([donor_dict[k] for k in matched_keys])

    # --------------------------------------------
    # 3) Compute concordance
    # --------------------------------------------
    concordance = pd.DataFrame(index=donor_samples, columns=bulk_samples, dtype=float)
    for d_idx, dname in enumerate(donor_samples):
        dvec = donor_gt_matrix[:, d_idx]
        for b_idx, bname in enumerate(bulk_samples):
            bvec = bulk_gt_matrix[:, b_idx]
            mask = ~np.isnan(dvec) & ~np.isnan(bvec)
            concordance.loc[dname, bname] = (
                np.sum(dvec[mask] == bvec[mask]) / mask.sum() if mask.sum() > 0 else np.nan
            )
    concordance.to_csv(os.path.join(outdir, "donor_bulk_concordance.tsv"), sep="\t")

    # --------------------------------------------
    # 4) Donor → Bulk mapping
    # --------------------------------------------
    mapping_records = []
    donor_to_bulk = {}
    for donor in donor_samples:
        row = concordance.loc[donor].astype(float)
        if row.isna().all():
            mapping_records.append((donor, None, np.nan, None, np.nan, "rejected"))
            donor_to_bulk[donor] = None
            continue

        best_bulk = row.idxmax()
        best_val = row.max()
        sorted_vals = row.sort_values(ascending=False)
        second_val = sorted_vals.iloc[1] if len(sorted_vals) > 1 else 0
        second_bulk = sorted_vals.index[1] if len(sorted_vals) > 1 else None

        if best_val < min_concordance:
            status = "rejected"
            donor_to_bulk[donor] = None
        elif (best_val - second_val) <= ambiguity_delta:
            status = "ambiguous"
            donor_to_bulk[donor] = best_bulk
        else:
            status = "mapped"
            donor_to_bulk[donor] = best_bulk

        mapping_records.append((donor, best_bulk, best_val, second_bulk, second_val, status))

    mapping_df = pd.DataFrame(
        mapping_records,
        columns=["donor", "best_bulk", "best_val", "second_bulk", "second_val", "status"]
    )
    mapping_df.to_csv(os.path.join(outdir, "donor_to_bulk_mapping.tsv"), sep="\t", index=False)

    # --------------------------------------------
    # 5) Combined demux results
    # --------------------------------------------
    combined_df = pd.read_csv(combined_tsv, sep="\t")
    combined_df["BulkSample"] = combined_df["Vireo_Individual_Assignment"].map(donor_to_bulk)
    combined_df["MappingStatus"] = combined_df.apply(
        lambda r: "doublet" if r["MajoritySinglet_DropletType"] == "doublet"
        else "unassigned" if r["MajoritySinglet_Individual_Assignment"] in ["doublet", "unassigned"]
        else "singlet", axis=1
    )
    enriched_path = os.path.join(outdir, "combined_results_with_bulk_mapping.tsv")
    combined_df.to_csv(enriched_path, sep="\t", index=False)

    # --------------------------------------------
    # ✅ Mapping summary
    # --------------------------------------------
    mapped_count = (mapping_df["status"] == "mapped").sum()
    ambiguous_count = (mapping_df["status"] == "ambiguous").sum()
    rejected_count = (mapping_df["status"] == "rejected").sum()
    unique_bulk = mapping_df["best_bulk"].dropna().unique().tolist()

    high_conf_df = combined_df[
        (combined_df["MappingStatus"] == "singlet") &
        (~combined_df["BulkSample"].isna())
    ].copy()
    cell_counts = high_conf_df["BulkSample"].value_counts().to_dict()

    summary_lines = [
        f"### Mapping Summary — {multiplex_name or ''}",
        f"Total donors: {len(mapping_df)}",
        f"Mapped donors: {mapped_count}",
        f"Ambiguous donors: {ambiguous_count}",
        f"Rejected donors: {rejected_count}",
        f"Unique bulk samples mapped: {len(unique_bulk)} → {', '.join(unique_bulk)}",
        "",
        "Cells per mapped bulk sample:",
    ] + [f"  {b}: {n} cells" for b, n in cell_counts.items()] + [
        "",
        "Detailed donor-to-bulk mapping table:",
        mapping_df.to_string(index=False)
    ]

    summary_path = os.path.join(outdir, "mapping_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    # --------------------------------------------
    # 6) Consensus heatmap data
    # --------------------------------------------
    high_conf_df["Vireo_vs_Souporcell"] = (
        high_conf_df["Vireo_Individual_Assignment"].astype(str)
        + " | "
        + high_conf_df["Souporcell_Individual_Assignment"].astype(str)
    )
    counts = high_conf_df.groupby(["Vireo_vs_Souporcell", "BulkSample"]).size().reset_index(name="CellCount")
    heatmap_df = counts.pivot(index="Vireo_vs_Souporcell", columns="BulkSample", values="CellCount").fillna(0)
    fractions = heatmap_df.div(heatmap_df.sum(axis=1), axis=0).fillna(0)
    row_order = fractions.idxmax(axis=1).sort_values().index
    col_order = fractions.sum(axis=0).sort_values(ascending=False).index
    fra_mat = fractions.loc[row_order, col_order]
    count_mat = heatmap_df.loc[row_order, col_order]

    # --------------------------------------------
    # 7) Geno Prob Δ heatmap
    # --------------------------------------------
    try:
        import vireoSNP
        res = vireoSNP.vcf.match_VCF_samples(donor_vcf, bulk_vcf, GT_tag1="GT", GT_tag2="GT")
        matched_GPb_diff = np.array(res["matched_GPb_diff"])
        matched_donors1 = res["matched_donors1"]
        matched_donors2 = res["matched_donors2"]
        mapped_idx = [i for i, d in enumerate(matched_donors1) if donor_to_bulk.get(d) is not None]
        if mapped_idx:
            matched_GPb_diff = matched_GPb_diff[mapped_idx, :]
            matched_donors1 = [matched_donors1[i] for i in mapped_idx]
            mapped_bulk = set(donor_to_bulk.values())
            cols_to_keep = [i for i, b in enumerate(matched_donors2) if b in mapped_bulk]
            if cols_to_keep:
                matched_GPb_diff = matched_GPb_diff[:, cols_to_keep]
                matched_donors2 = [matched_donors2[i] for i in cols_to_keep]
            else:
                matched_GPb_diff = None
        else:
            matched_GPb_diff = None
    except Exception:
        matched_GPb_diff = None

    # --------------------------------------------
    # 8) Show and save combined figure
    # --------------------------------------------
    
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6), gridspec_kw={'wspace': 0.20})
    # LEFT PANEL: Consensus heatmap
    ax1 = axes[0]
    heatmap1 = sns.heatmap(
        fra_mat, annot=count_mat.astype(int), fmt="d",
        cmap=cmap, linewidths=0.5, linecolor=border_color,
        ax=ax1, cbar_kws={'label': "Fraction of cells", 'orientation': 'vertical'}
    )
    cbar1 = heatmap1.collections[0].colorbar
    cbar1.ax.yaxis.set_ticks_position('right')
    cbar1.ax.yaxis.set_label_position('left')
    cbar1.ax.yaxis.set_label_coords(-0.3, 0.5)
    cbar1.set_label("Fraction of cells", rotation=90, labelpad=100,
                    fontweight='bold', fontsize=12)
    ax1.set_title(f"{multiplex_name} — Consensus Assignments", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Bulk Samples", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Vireo | Souporcell", fontsize=13, fontweight="bold")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, fontsize=13)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=13)

    # RIGHT PANEL: Geno Prob Δ heatmap
    ax2 = axes[1]
    if matched_GPb_diff is not None:
        heatmap2 = sns.heatmap(
            matched_GPb_diff, xticklabels=matched_donors2, yticklabels=matched_donors1,
            annot=np.round(matched_GPb_diff, 2), fmt=".2f",
            cmap=cmap, linewidths=0.5, linecolor=border_color,
            ax=ax2, cbar_kws={'label': "Geno Prob Δ", 'orientation': 'vertical'}
        )
        cbar2 = heatmap2.collections[0].colorbar
        cbar2.ax.yaxis.set_ticks_position('right')
        cbar2.ax.yaxis.set_label_position('left')
        cbar2.ax.yaxis.set_label_coords(-0.3, 0.5)
        cbar2.set_label("Geno Prob Δ", rotation=90, labelpad=100,
                        fontweight='bold', fontsize=12)
        n_snps = res.get("matched_n_var", matched_GPb_diff.shape[1])
        ax2.set_title(f"{multiplex_name} — Geno Prob Delta — {n_snps} SNPs", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Bulk Samples", fontsize=13, fontweight="bold")
        ax2.set_ylabel("Vireo Donors", fontsize=13, fontweight="bold")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, fontsize=13)
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=13)
    else:
        ax2.text(0.5, 0.5, "No SNPs available", ha="center", va="center", fontsize=13)
        ax2.axis("off")

    plt.tight_layout()
    main_pdf = os.path.join(outdir, "MainFigure_TwoColumn.pdf")
    main_png = os.path.join(outdir, "MainFigure_TwoColumn.png")
    fig.savefig(main_pdf)
    fig.savefig(main_png, dpi=200)
    plt.show()


    
    # --------------------------------------------
    # 9) Print summary
    # --------------------------------------------
    print("\n========== MAPPING SUMMARY ==========")
    print(mapping_df.to_string(index=False))
    print("=====================================")
    print(f"✅ Saved mapping summary: {summary_path}")
    print(f"✅ Saved main figure: {main_pdf}")
    print(f"✅ Saved PNG figure: {main_png}\n")

    return combined_df, mapping_df



run_bulk_donor_mapping(
    bulk_vcf=bulk_vcf,
    donor_vcf=FH.S1256_donor_vcf,
    combined_tsv=FH.S1256_combined_tsv,
    outdir=FH.S1256_outdir,
    min_concordance=0.6,
    ambiguity_delta=0.05,
    multiplex_name="FH.S1256"
)



combined_df, mapping_df = run_bulk_donor_mapping(
    bulk_vcf=bulk_vcf,
    donor_vcf=FH.S3478_donor_vcf,
    combined_tsv=FH.S3478_combined_tsv,
    outdir=FH.S3478_outdir,
    min_concordance=0.6,
    ambiguity_delta=0.05,
    multiplex_name="FH.S3478"
)
