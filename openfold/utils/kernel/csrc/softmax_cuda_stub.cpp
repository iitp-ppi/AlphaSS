{"payload":{"allShortcutsEnabled":false,"fileTree":{"openfold/utils/kernel/csrc":{"items":[{"name":"compat.h","path":"openfold/utils/kernel/csrc/compat.h","contentType":"file"},{"name":"softmax_cuda.cpp","path":"openfold/utils/kernel/csrc/softmax_cuda.cpp","contentType":"file"},{"name":"softmax_cuda_kernel.cu","path":"openfold/utils/kernel/csrc/softmax_cuda_kernel.cu","contentType":"file"},{"name":"softmax_cuda_stub.cpp","path":"openfold/utils/kernel/csrc/softmax_cuda_stub.cpp","contentType":"file"}],"totalCount":4},"openfold/utils/kernel":{"items":[{"name":"csrc","path":"openfold/utils/kernel/csrc","contentType":"directory"},{"name":"__init__.py","path":"openfold/utils/kernel/__init__.py","contentType":"file"},{"name":"attention_core.py","path":"openfold/utils/kernel/attention_core.py","contentType":"file"}],"totalCount":3},"openfold/utils":{"items":[{"name":"kernel","path":"openfold/utils/kernel","contentType":"directory"},{"name":"__init__.py","path":"openfold/utils/__init__.py","contentType":"file"},{"name":"argparse.py","path":"openfold/utils/argparse.py","contentType":"file"},{"name":"callbacks.py","path":"openfold/utils/callbacks.py","contentType":"file"},{"name":"checkpointing.py","path":"openfold/utils/checkpointing.py","contentType":"file"},{"name":"chunk_utils.py","path":"openfold/utils/chunk_utils.py","contentType":"file"},{"name":"exponential_moving_average.py","path":"openfold/utils/exponential_moving_average.py","contentType":"file"},{"name":"feats.py","path":"openfold/utils/feats.py","contentType":"file"},{"name":"import_weights.py","path":"openfold/utils/import_weights.py","contentType":"file"},{"name":"logger.py","path":"openfold/utils/logger.py","contentType":"file"},{"name":"loss.py","path":"openfold/utils/loss.py","contentType":"file"},{"name":"lr_schedulers.py","path":"openfold/utils/lr_schedulers.py","contentType":"file"},{"name":"precision_utils.py","path":"openfold/utils/precision_utils.py","contentType":"file"},{"name":"rigid_utils.py","path":"openfold/utils/rigid_utils.py","contentType":"file"},{"name":"script_utils.py","path":"openfold/utils/script_utils.py","contentType":"file"},{"name":"seed.py","path":"openfold/utils/seed.py","contentType":"file"},{"name":"superimposition.py","path":"openfold/utils/superimposition.py","contentType":"file"},{"name":"suppress_output.py","path":"openfold/utils/suppress_output.py","contentType":"file"},{"name":"tensor_utils.py","path":"openfold/utils/tensor_utils.py","contentType":"file"},{"name":"trace_utils.py","path":"openfold/utils/trace_utils.py","contentType":"file"},{"name":"validation_metrics.py","path":"openfold/utils/validation_metrics.py","contentType":"file"}],"totalCount":21},"openfold":{"items":[{"name":"data","path":"openfold/data","contentType":"directory"},{"name":"model","path":"openfold/model","contentType":"directory"},{"name":"np","path":"openfold/np","contentType":"directory"},{"name":"resources","path":"openfold/resources","contentType":"directory"},{"name":"utils","path":"openfold/utils","contentType":"directory"},{"name":"__init__.py","path":"openfold/__init__.py","contentType":"file"},{"name":"config.py","path":"openfold/config.py","contentType":"file"}],"totalCount":7},"":{"items":[{"name":".github","path":".github","contentType":"directory"},{"name":"imgs","path":"imgs","contentType":"directory"},{"name":"lib","path":"lib","contentType":"directory"},{"name":"notebooks","path":"notebooks","contentType":"directory"},{"name":"openfold","path":"openfold","contentType":"directory"},{"name":"scripts","path":"scripts","contentType":"directory"},{"name":"tests","path":"tests","contentType":"directory"},{"name":".gitignore","path":".gitignore","contentType":"file"},{"name":"CITATION.cff","path":"CITATION.cff","contentType":"file"},{"name":"Dockerfile","path":"Dockerfile","contentType":"file"},{"name":"LICENSE","path":"LICENSE","contentType":"file"},{"name":"README.md","path":"README.md","contentType":"file"},{"name":"deepspeed_config.json","path":"deepspeed_config.json","contentType":"file"},{"name":"environment.yml","path":"environment.yml","contentType":"file"},{"name":"run_pretrained_openfold.py","path":"run_pretrained_openfold.py","contentType":"file"},{"name":"setup.py","path":"setup.py","contentType":"file"},{"name":"thread_sequence.py","path":"thread_sequence.py","contentType":"file"},{"name":"train_openfold.py","path":"train_openfold.py","contentType":"file"}],"totalCount":18}},"fileTreeProcessingTime":12.347506,"foldersToFetch":[],"reducedMotionEnabled":null,"repo":{"id":406555580,"defaultBranch":"main","name":"openfold","ownerLogin":"aqlaboratory","currentUserCanPush":false,"isFork":false,"isEmpty":false,"createdAt":"2021-09-14T23:59:02.000Z","ownerAvatar":"https://avatars.githubusercontent.com/u/8396911?v=4","public":true,"private":false,"isOrgOwned":true},"symbolsExpanded":false,"treeExpanded":true,"refInfo":{"name":"main","listCacheKey":"v0:1695073707.0","canEdit":false,"refType":"branch","currentOid":"2134cc09b3994b6280e6e3c569dd7d761e4da7a0"},"path":"openfold/utils/kernel/csrc/softmax_cuda_stub.cpp","currentUser":null,"blob":{"rawLines":["// Copyright 2021 AlQuraishi Laboratory","//","// Licensed under the Apache License, Version 2.0 (the \"License\");","// you may not use this file except in compliance with the License.","// You may obtain a copy of the License at","//","//      http://www.apache.org/licenses/LICENSE-2.0","//","// Unless required by applicable law or agreed to in writing, software","// distributed under the License is distributed on an \"AS IS\" BASIS,","// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.","// See the License for the specific language governing permissions and","// limitations under the License.","","// modified from fastfold/model/fastnn/kernel/cuda_native/csrc/softmax_cuda.cpp","","#include <torch/extension.h>","","void attn_softmax_inplace_forward_(","    at::Tensor input, ","    long long rows, int cols",")","{","    throw std::runtime_error(\"attn_softmax_inplace_forward_ not implemented on CPU\");","};","void attn_softmax_inplace_backward_(","    at::Tensor output, ","    at::Tensor d_ov,","    at::Tensor values,","    long long rows, ","    int cols_output,","    int cols_values",")","{","    throw std::runtime_error(\"attn_softmax_inplace_backward_ not implemented on CPU\");","};"],"stylingDirectives":[[{"start":0,"end":39,"cssClass":"pl-c"},{"start":0,"end":2,"cssClass":"pl-c"}],[{"start":0,"end":2,"cssClass":"pl-c"},{"start":0,"end":2,"cssClass":"pl-c"}],[{"start":0,"end":66,"cssClass":"pl-c"},{"start":0,"end":2,"cssClass":"pl-c"}],[{"start":0,"end":67,"cssClass":"pl-c"},{"start":0,"end":2,"cssClass":"pl-c"}],[{"start":0,"end":42,"cssClass":"pl-c"},{"start":0,"end":2,"cssClass":"pl-c"}],[{"start":0,"end":2,"cssClass":"pl-c"},{"start":0,"end":2,"cssClass":"pl-c"}],[{"start":0,"end":50,"cssClass":"pl-c"},{"start":0,"end":2,"cssClass":"pl-c"}],[{"start":0,"end":2,"cssClass":"pl-c"},{"start":0,"end":2,"cssClass":"pl-c"}],[{"start":0,"end":70,"cssClass":"pl-c"},{"start":0,"end":2,"cssClass":"pl-c"}],[{"start":0,"end":68,"cssClass":"pl-c"},{"start":0,"end":2,"cssClass":"pl-c"}],[{"start":0,"end":75,"cssClass":"pl-c"},{"start":0,"end":2,"cssClass":"pl-c"}],[{"start":0,"end":70,"cssClass":"pl-c"},{"start":0,"end":2,"cssClass":"pl-c"}],[{"start":0,"end":33,"cssClass":"pl-c"},{"start":0,"end":2,"cssClass":"pl-c"}],[],[{"start":0,"end":79,"cssClass":"pl-c"},{"start":0,"end":2,"cssClass":"pl-c"}],[],[{"start":1,"end":8,"cssClass":"pl-k"},{"start":9,"end":28,"cssClass":"pl-s"},{"start":9,"end":10,"cssClass":"pl-pds"},{"start":27,"end":28,"cssClass":"pl-pds"}],[],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":34,"cssClass":"pl-en"}],[],[{"start":4,"end":8,"cssClass":"pl-k"},{"start":9,"end":13,"cssClass":"pl-k"},{"start":20,"end":23,"cssClass":"pl-k"}],[],[],[{"start":4,"end":9,"cssClass":"pl-k"},{"start":10,"end":28,"cssClass":"pl-smi"},{"start":29,"end":83,"cssClass":"pl-s"},{"start":29,"end":30,"cssClass":"pl-pds"},{"start":82,"end":83,"cssClass":"pl-pds"}],[],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":35,"cssClass":"pl-en"}],[],[],[],[{"start":4,"end":8,"cssClass":"pl-k"},{"start":9,"end":13,"cssClass":"pl-k"}],[{"start":4,"end":7,"cssClass":"pl-k"}],[{"start":4,"end":7,"cssClass":"pl-k"}],[],[],[{"start":4,"end":9,"cssClass":"pl-k"},{"start":10,"end":28,"cssClass":"pl-smi"},{"start":29,"end":84,"cssClass":"pl-s"},{"start":29,"end":30,"cssClass":"pl-pds"},{"start":83,"end":84,"cssClass":"pl-pds"}],[]],"csv":null,"csvError":null,"dependabotInfo":{"showConfigurationBanner":false,"configFilePath":null,"networkDependabotPath":"/aqlaboratory/openfold/network/updates","dismissConfigurationNoticePath":"/settings/dismiss-notice/dependabot_configuration_notice","configurationNoticeDismissed":null,"repoAlertsPath":"/aqlaboratory/openfold/security/dependabot","repoSecurityAndAnalysisPath":"/aqlaboratory/openfold/settings/security_analysis","repoOwnerIsOrg":true,"currentUserCanAdminRepo":false},"displayName":"softmax_cuda_stub.cpp","displayUrl":"https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/kernel/csrc/softmax_cuda_stub.cpp?raw=true","headerInfo":{"blobSize":"1.13 KB","deleteInfo":{"deleteTooltip":"You must be signed in to make or propose changes"},"editInfo":{"editTooltip":"You must be signed in to make or propose changes"},"ghDesktopPath":"https://desktop.github.com","gitLfsPath":null,"onBranch":true,"shortPath":"4539c19","siteNavLoginPath":"/login?return_to=https%3A%2F%2Fgithub.com%2Faqlaboratory%2Fopenfold%2Fblob%2Fmain%2Fopenfold%2Futils%2Fkernel%2Fcsrc%2Fsoftmax_cuda_stub.cpp","isCSV":false,"isRichtext":false,"toc":null,"lineInfo":{"truncatedLoc":"36","truncatedSloc":"33"},"mode":"file"},"image":false,"isCodeownersFile":null,"isPlain":false,"isValidLegacyIssueTemplate":false,"issueTemplateHelpUrl":"https://docs.github.com/articles/about-issue-and-pull-request-templates","issueTemplate":null,"discussionTemplate":null,"language":"C++","languageID":43,"large":false,"loggedIn":false,"newDiscussionPath":"/aqlaboratory/openfold/discussions/new","newIssuePath":"/aqlaboratory/openfold/issues/new","planSupportInfo":{"repoIsFork":null,"repoOwnedByCurrentUser":null,"requestFullPath":"/aqlaboratory/openfold/blob/main/openfold/utils/kernel/csrc/softmax_cuda_stub.cpp","showFreeOrgGatedFeatureMessage":null,"showPlanSupportBanner":null,"upgradeDataAttributes":null,"upgradePath":null},"publishBannersInfo":{"dismissActionNoticePath":"/settings/dismiss-notice/publish_action_from_dockerfile","dismissStackNoticePath":"/settings/dismiss-notice/publish_stack_from_file","releasePath":"/aqlaboratory/openfold/releases/new?marketplace=true","showPublishActionBanner":false,"showPublishStackBanner":false},"renderImageOrRaw":false,"richText":null,"renderedFileInfo":null,"shortPath":null,"tabSize":8,"topBannersInfo":{"overridingGlobalFundingFile":false,"globalPreferredFundingPath":null,"repoOwner":"aqlaboratory","repoName":"openfold","showInvalidCitationWarning":false,"citationHelpUrl":"https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/about-citation-files","showDependabotConfigurationBanner":false,"actionsOnboardingTip":null},"truncated":false,"viewable":true,"workflowRedirectUrl":null,"symbols":{"timedOut":false,"notAnalyzed":false,"symbols":[{"name":"attn_softmax_inplace_forward_","kind":"function","identStart":716,"identEnd":745,"extentStart":716,"extentEnd":800,"fullyQualifiedName":"attn_softmax_inplace_forward_","identUtf16":{"start":{"lineNumber":18,"utf16Col":5},"end":{"lineNumber":18,"utf16Col":34}},"extentUtf16":{"start":{"lineNumber":18,"utf16Col":5},"end":{"lineNumber":21,"utf16Col":1}}},{"name":"attn_softmax_inplace_backward_","kind":"function","identStart":897,"identEnd":927,"extentStart":897,"extentEnd":1060,"fullyQualifiedName":"attn_softmax_inplace_backward_","identUtf16":{"start":{"lineNumber":25,"utf16Col":5},"end":{"lineNumber":25,"utf16Col":35}},"extentUtf16":{"start":{"lineNumber":25,"utf16Col":5},"end":{"lineNumber":32,"utf16Col":1}}}]}},"copilotInfo":null,"csrf_tokens":{"/aqlaboratory/openfold/branches":{"post":"jK_hNpf5DWLwcvQZ_BOvgsEK7Hl-YDfArEo4RWXJlba56OUuEUGj8Ew_vNumF_8SS04JJrjBVrkO3q9mWITXwg"},"/repos/preferences":{"post":"iv_jM-gjJ4Wi4YAx3Qk9mNcl1MyyHImOFCiB5pdLUWf44YJ7gbTHmcouOX88fbaAIqBhjEaH2DPcFf60a11IxQ"}}},"title":"openfold/openfold/utils/kernel/csrc/softmax_cuda_stub.cpp at main · aqlaboratory/openfold"}