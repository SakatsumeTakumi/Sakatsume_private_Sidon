#!/usr/bin/env bash
set -euo pipefail

for lang in ab af am ar as ast az ba bas be bg bn br ca ckb cnh cs cv cy da dav de dv dyu el en eo es et eu fa fi fr fy-NL ga-IE gl gn ha he hi hsb ht hu hy-AM ia id ig is it ja ka kab kk kln kmr ko ky lg lij lo lt ltg luo lv mdf mhr mk ml mn mr mrj mt myv nan-tw nb-NO ne-NP nhi nl nn-NO nr nso oc or os pa-IN pl ps pt quy rm-sursilv rm-vallader ro ru rup rw sah sat sc sd sk skr sl sq sr st sv-SE sw ta te tg th ti tig tk tn tok tr ts tt tw ug uk ur uz ve vi vot xh yi yo yue zgh zh-CN zh-HK zh-TW zu zza; do
  # set extra qsub opts for heavy languages
  declare -a extra_qsub_opts=()
  case "$lang" in
    en|bn|es) extra_qsub_opts=(-l walltime=24:00:00) ;;
  esac
  for split in train validation test other invalidated; do
    done_flag="/groups/gcb50354/common_voice22_sidon/$lang/completed_${split}.txt"
    if [[ -f "$done_flag" ]]; then
      :
    else
      # Quote the -v argument; hyphens in values are fine for PBS.
      echo "submitting $split for $lang"
      qsub "${extra_qsub_opts[@]}" \
      -v "MLS_LANGUAGE=${lang},SPLIT=${split}" \
      scripts/pbs/cleanse/common_voice/commonvoice.sh
    fi
  done
done

