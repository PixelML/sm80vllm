# Vendored baseline

The 48 files in `vllm/` are the **pre-patch** copies of every file that
`promisezackr/glm53-flash-170hx-pp8` modifies, taken verbatim out of the
official image:

    vllm/vllm-openai:glm53-flash
    digest sha256:2c6da6c6f16ed15c91e412d896dba13701f25fe1861eaec9ddaa4db34d1d21c4
    vLLM 0.1.dev20051+g487ecf187

Upstream commit `487ecf187` is **not public** - it is not in vllm-project/vllm,
not in the GLM release PR head, and not reachable from any remote we have. It
exists only inside the published image. That is why this branch is an orphan
rooted at a vendored file set rather than a rebase.

License: Apache-2.0 (the vLLM project). Files are unmodified in this commit;
every subsequent commit is one patch from the source repo, attributed.
