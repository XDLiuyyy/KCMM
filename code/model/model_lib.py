import torch
import torch.nn as nn
import numpy as np
# from model.resnet3d_xl import Net
from model.swin_modeling.load_swin import Net
# from model.load_mvit import Net
from torchvision.ops.roi_align import roi_align
import re
import random

from transformers import BertPreTrainedModel, BertModel, AlbertConfig
from transformers import RobertaConfig, RobertaModel
from transformers.modeling_roberta import RobertaLMHead
import json

DEVICE = torch.device("cuda:1")


def get_features_from_task_1_with_kbert(context, tokenizer, max_seq_length, graph):
    context_tokens = str(context)
    context_tokens = context_tokens.lower()
    source_sent = '{} {} {}'.format(tokenizer.cls_token,
                                    context_tokens,
                                    tokenizer.sep_token)
    tokens, soft_pos_id, attention_mask, segment_ids = add_knowledge_with_vm(mp_all=graph.mp_all,
                                                                             sent_batch=[source_sent],
                                                                             tokenizer=tokenizer,
                                                                             max_entities=3,
                                                                             max_length=max_seq_length)

    tokens = tokens[0]
    soft_pos_id = torch.tensor(soft_pos_id[0])
    attention_mask = torch.tensor(attention_mask[0])

    segment_ids = torch.tensor(segment_ids[0])
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))

    assert input_ids.shape[0] == max_seq_length
    assert attention_mask.shape[0] == max_seq_length
    assert soft_pos_id.shape[0] == max_seq_length
    assert segment_ids.shape[0] == max_seq_length

    if 'Roberta' in str(type(tokenizer)):
        soft_pos_id = soft_pos_id + 2

    choices_features = (tokens, input_ids, attention_mask, segment_ids, soft_pos_id)

    return choices_features


def add_knowledge_with_vm(mp_all,
                          sent_batch,
                          tokenizer,
                          max_entities=2,
                          max_length=128):
    def conceptnet_relation_to_nl(ent):
        relation_to_language = {
            '/r/ReceivesAction': 'receives action for',
            '/r/RelatedTo': 'is related to',
            '/r/UsedFor': 'is used for',
        }

        ent_values = 'i {} {}'.format(relation_to_language.get(ent[1], ''),
                                      ent[0].replace('_', ' '))
        ent_values = tokenizer.tokenize(ent_values)[1:]
        return ent_values

    split_sent_batch = [tokenizer.tokenize(sent) for sent in sent_batch]

    know_sent_batch = []
    position_batch = []
    visible_matrix_batch = []
    seg_batch = []

    for split_sent in split_sent_batch:

        sent_tree = []
        pos_idx_tree = []
        abs_idx_tree = []
        pos_idx = -1
        abs_idx = -1
        abs_idx_src = []
        for token in split_sent:

            entities = sorted(list(mp_all.get(token.strip(',|.|?|;|:|!|Ġ|_|▁'), [])), key=lambda x: x[2], reverse=True)[
                       :max_entities]

            sent_tree.append((token, entities))

            if token in tokenizer.all_special_tokens:
                token_pos_idx = [pos_idx + 1]
                token_abs_idx = [abs_idx + 1]
            else:
                token_pos_idx = [pos_idx + 1]
                token_abs_idx = [abs_idx + 1]
            abs_idx = token_abs_idx[-1]

            entities_pos_idx = []
            entities_abs_idx = []
            for ent in entities:
                ent_values = conceptnet_relation_to_nl(ent)

                ent_pos_idx = [
                    token_pos_idx[-1] + i for i in range(1,
                                                         len(ent_values) + 1)
                ]
                entities_pos_idx.append(ent_pos_idx)
                ent_abs_idx = [abs_idx + i for i in range(1, len(ent_values) + 1)]
                abs_idx = ent_abs_idx[-1]
                entities_abs_idx.append(ent_abs_idx)

            pos_idx_tree.append((token_pos_idx, entities_pos_idx))
            pos_idx = token_pos_idx[-1]
            abs_idx_tree.append((token_abs_idx, entities_abs_idx))
            abs_idx_src += token_abs_idx

        know_sent = []
        pos = []
        seg = []
        for i in range(len(sent_tree)):
            word = sent_tree[i][0]
            if word in tokenizer.all_special_tokens:
                know_sent += [word]
                seg += [0]
            else:
                know_sent += [word]
                seg += [0]
            pos += pos_idx_tree[i][0]
            for j in range(len(sent_tree[i][1])):
                ent = sent_tree[i][1][j]
                ent_values = conceptnet_relation_to_nl(ent)

                add_word = ent_values
                know_sent += add_word
                seg += [1] * len(add_word)
                pos += list(pos_idx_tree[i][1][j])

        token_num = len(know_sent)

        visible_matrix = np.zeros((token_num, token_num))
        for item in abs_idx_tree:
            src_ids = item[0]
            for id in src_ids:
                visible_abs_idx = abs_idx_src + [
                    idx for ent in item[1] for idx in ent
                ]
                visible_matrix[id, visible_abs_idx] = 1

            for ent in item[1]:
                for id in ent:
                    visible_abs_idx = ent + src_ids
                    visible_matrix[id, visible_abs_idx] = 1

        src_length = len(know_sent)
        if len(know_sent) < max_length:
            pad_num = max_length - src_length
            know_sent += [tokenizer.pad_token] * pad_num
            seg += [0] * pad_num
            pos += [max_length - 1] * pad_num
            visible_matrix = np.pad(visible_matrix,
                                    ((0, pad_num), (0, pad_num)),
                                    'constant')
        else:
            know_sent = know_sent[:max_length]
            seg = seg[:max_length]
            pos = pos[:max_length]
            visible_matrix = visible_matrix[:max_length, :max_length]

        know_sent_batch.append(know_sent)
        position_batch.append(pos)
        visible_matrix_batch.append(visible_matrix)
        seg_batch.append(seg)

    return know_sent_batch, position_batch, visible_matrix_batch, seg_batch


def process_context(tokenizer, context_tokens, model, graph):
    tokens_tmp = []
    model.to(DEVICE)

    for i in range(len(context_tokens)):
        input_token = get_features_from_task_1_with_kbert(context_tokens[i], tokenizer, 128, graph)
        tokens_tmp.append(input_token)

    x = [tokens_tmp]

    num_choices = len(x[0])
    input_ids = torch.stack([j[1] for i in x for j in i], dim=0).reshape(
        (-1, num_choices,) + x[0][0][1].shape).to(DEVICE)
    attention_mask = torch.stack([j[2] for i in x for j in i], dim=0).reshape(
        (-1, num_choices,) + x[0][0][2].shape).to(DEVICE)
    token_type_ids = torch.stack([j[3] for i in x for j in i], dim=0).reshape(
        (-1, num_choices,) + x[0][0][3].shape).to(DEVICE)
    position_ids = torch.stack([j[4] for i in x for j in i], dim=0).reshape(
        (-1, num_choices,) + x[0][0][4].shape).to(DEVICE)

    output = model(input_ids=input_ids,
                   attention_mask=attention_mask,
                   position_ids=position_ids,
                   labels=None)
    return output


class RobertaForMultipleChoiceWithLM(BertPreTrainedModel):
    r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            To match pre-training, RoBerta input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] [SEP] no it is not . [SEP]``

                ``token_type_ids:   0   0  0    0    0     0       0   0   0     1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``

                ``token_type_ids:   0   0   0   0  0     0   0``

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, num_choices, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            The second dimension of the input (`num_choices`) indicates the number of choices to score.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMultipleChoice.from_pretrained('roberta-base')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s, add_special_tokens=True) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = {
        'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
        'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
        'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    }
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForMultipleChoiceWithLM, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lamda1 = nn.Parameter(torch.rand(1) * 2 + 1)
        self.lamda2 = nn.Parameter(torch.rand(1) * 2 + 1)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        if attention_mask.dim() == 3:
            flat_attention_mask = attention_mask.view(-1,
                                                      attention_mask.size(-1)) if attention_mask is not None else None
        else:
            flat_attention_mask = attention_mask.view(
                (-1,) + attention_mask.shape[-2:]) if attention_mask is not None else None
        outputs = self.roberta(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids,
                               attention_mask=flat_attention_mask, head_mask=head_mask)

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        outputs = (reshaped_logits,
                   pooled_output.view(input_ids.shape[0], num_choices, -1),) + outputs[
                                                                               2:]

        return outputs


def box_to_normalized(boxes_tensor, crop_size=[224, 224], mode='list'):
    # tensor to list, and [cx, cy, w, h] --> [x1, y1, x2, y2]
    new_boxes_tensor = boxes_tensor.clone()
    new_boxes_tensor[..., 0] = (
                                       boxes_tensor[..., 0] - boxes_tensor[..., 2] / 2.0) * crop_size[0]
    new_boxes_tensor[..., 1] = (
                                       boxes_tensor[..., 1] - boxes_tensor[..., 3] / 2.0) * crop_size[1]
    new_boxes_tensor[..., 2] = (
                                       boxes_tensor[..., 0] + boxes_tensor[..., 2] / 2.0) * crop_size[0]
    new_boxes_tensor[..., 3] = (
                                       boxes_tensor[..., 1] + boxes_tensor[..., 3] / 2.0) * crop_size[1]
    if mode == 'list':
        boxes_list = []
        for boxes in new_boxes_tensor:
            boxes_list.append(boxes)
        return boxes_list
    elif mode == 'tensor':
        return new_boxes_tensor


def build_region_feas(feature_maps, boxes_list, output_crop_size=[3, 3], img_size=[224, 224]):
    # Building feas for each bounding box by using RoI Align
    # feature_maps:[N,C,H,W], where N=b*T
    IH, IW = img_size
    FH, FW = feature_maps.size()[-2:]  # Feature_H, Feature_W
    region_feas = roi_align(feature_maps, boxes_list, output_crop_size,
                            spatial_scale=float(FW) / IW)  # b*T*K, C, S, S; S denotes output_size
    return region_feas.view(region_feas.size(0), -1)  # b*T*K, D*S*S


class BboxVisualModel(nn.Module):
    '''
    backbone: swin
    '''

    def __init__(self, opt):
        nn.Module.__init__(self)
        self.nr_actions = opt.num_classes
        self.nr_frames = opt.num_frames
        self.nr_boxes = opt.num_boxes

        self.img_feature_dim = 512
        self.backbone = Net()
        # swin 1024
        # self.conv = nn.Conv2d(1024, self.img_feature_dim, kernel_size=(1, 1), stride=1)
        self.conv = nn.Conv2d(768, self.img_feature_dim, kernel_size=(1, 1), stride=1)
        self.crop_size = [3, 3]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool1d = nn.AdaptiveAvgPool1d(512)

        self.dropout = nn.Dropout(0.3)

        self.region_vis_embed = nn.Sequential(
            nn.Linear(self.img_feature_dim * self.crop_size[0] * self.crop_size[1], self.img_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        self.fc = nn.Linear(512, self.nr_actions)
        self.com_fc = nn.Linear(512, self.nr_actions)

        self.count = 0

        with open(
                "/home/ly/CAction/KCMM/data/dataset_splits/compositional/object_All_new.json",
                encoding="utf-8") as f:
            self.object_json = json.load(f)
        with open(
                "/home/ly/CAction/KCMM/data/dataset_splits/compositional/labels.json",
                encoding="utf-8") as f:
            self.label_json = json.load(f)

        with open(
                "/home/ly/CAction/KCMM/data/dataset_splits/compositional/object_All_rewrite.json",
                encoding="utf-8") as f:
            self.train_all_object = json.load(f)

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not 'fc' in k and not 'classifier' in k:
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'fc' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'

    def combination(self, sub_num, obj_len, video1_label, object_label_list):

        obj_verb_sentence = []
        if sub_num == 1:
            if obj_len == 0:
                obj_verb_sentence = re.sub("something", "empty", video1_label, count=1,
                                           flags=re.IGNORECASE)
            else:
                obj_verb_sentence = re.sub("something", object_label_list[0], video1_label, count=1,
                                           flags=re.IGNORECASE)
        elif sub_num == 2:
            if obj_len == 0:
                obj_verb_sentence = re.sub("something", "empty", video1_label, count=1,
                                           flags=re.IGNORECASE)
                obj_verb_sentence = re.sub("something", "empty", obj_verb_sentence, count=1,
                                           flags=re.IGNORECASE)

            elif obj_len == 1:
                obj_verb_sentence = re.sub("something", object_label_list[0], video1_label, count=1,
                                           flags=re.IGNORECASE)
                obj_verb_sentence = re.sub("something", "empty", obj_verb_sentence, count=1, flags=re.IGNORECASE)
            else:
                obj_verb_sentence = re.sub("something", object_label_list[0], video1_label, count=1,
                                           flags=re.IGNORECASE)
                obj_verb_sentence = re.sub("something", object_label_list[1], obj_verb_sentence, count=1,
                                           flags=re.IGNORECASE)
        elif sub_num == 3:
            if obj_len == 0:
                obj_verb_sentence = re.sub("something", "empty", video1_label, count=1,
                                           flags=re.IGNORECASE)
                obj_verb_sentence = re.sub("something", "empty", obj_verb_sentence, count=1,
                                           flags=re.IGNORECASE)
                obj_verb_sentence = re.sub("something", "empty", obj_verb_sentence, count=1, flags=re.IGNORECASE)

            elif obj_len == 1:
                obj_verb_sentence = re.sub("something", object_label_list[0], video1_label, count=1,
                                           flags=re.IGNORECASE)
                obj_verb_sentence = re.sub("something", "empty", obj_verb_sentence, count=1, flags=re.IGNORECASE)
                obj_verb_sentence = re.sub("something", "empty", obj_verb_sentence, count=1, flags=re.IGNORECASE)

            elif obj_len == 2:
                obj_verb_sentence = re.sub("something", object_label_list[0], video1_label, count=1,
                                           flags=re.IGNORECASE)
                obj_verb_sentence = re.sub("something", object_label_list[1], obj_verb_sentence, count=1,
                                           flags=re.IGNORECASE)
                obj_verb_sentence = re.sub("something", "empty", obj_verb_sentence, count=1, flags=re.IGNORECASE)

            elif obj_len == 3:
                obj_verb_sentence = re.sub("something", object_label_list[0], video1_label, count=1,
                                           flags=re.IGNORECASE)
                obj_verb_sentence = re.sub("something", object_label_list[1], obj_verb_sentence, count=1,
                                           flags=re.IGNORECASE)
                obj_verb_sentence = re.sub("something", object_label_list[2], obj_verb_sentence, count=1,
                                           flags=re.IGNORECASE)
            else:
                obj_verb_sentence = re.sub("something", object_label_list[0], video1_label, count=1,
                                           flags=re.IGNORECASE)
                obj_verb_sentence = re.sub("something", object_label_list[1], obj_verb_sentence, count=1,
                                           flags=re.IGNORECASE)
                obj_verb_sentence = re.sub("something", object_label_list[2], obj_verb_sentence, count=1,
                                           flags=re.IGNORECASE)
        return obj_verb_sentence

    def forward(self, global_img_input, box_input, box_categories, video_label,
                gt_placeholders, verb_label,
                gt_placeholders_label_new_id, common_model, tokenizer, graph,
                is_inference=False):

        if is_inference:
            org_feas = self.backbone(global_img_input)  # (b, 2048, T/2, 14, 14)
            b, _, T, H, W = org_feas.size()
            org_feas = org_feas.permute(0, 2, 1, 3, 4).contiguous()
            # swin
            # org_feas = org_feas.view(b * T, 1024, H, W)
            org_feas = org_feas.view(b * T, 768, H, W)
            conv_fea_maps = self.conv(org_feas)  # (b*T, img_feature_dim)
            box_tensors = box_input.view(b * T, self.nr_boxes, 4)

            boxes_list = box_to_normalized(box_tensors, crop_size=[224, 224])
            img_size = global_img_input.size()[-2:]

            # (b*T*nr_boxes, C), C=3*3*d
            region_vis_feas = build_region_feas(conv_fea_maps, boxes_list, self.crop_size, img_size)

            region_vis_feas = self.region_vis_embed(region_vis_feas)

            region_vis_feas = region_vis_feas.view(b, T, self.nr_boxes,
                                                   region_vis_feas.size(-1))  # (b, t, n, img_feature_dim)
            region_vis_feas = region_vis_feas.permute(0, 3, 2, 1).contiguous()

            global_features = self.avgpool(region_vis_feas).squeeze()
            global_features = self.dropout(global_features)
            cls_output = self.fc(global_features)
            return cls_output, global_features
        else:

            video_label_list = []
            for key, value in self.label_json.items():
                for label_id in video_label:
                    if label_id == value:
                        video_label_list.append(key)

            video0_label = video_label_list[0]
            video1_label = video_label_list[1]
            video2_label = video_label_list[2]
            video3_label = video_label_list[3]

            gt_placeholders_label_new_id = gt_placeholders_label_new_id.tolist()

            new_key_list = list(self.train_all_object.keys())
            new_values_list = list(self.train_all_object.values())

            new_object_label_list0 = []
            new_object_label_list1 = []
            new_object_label_list2 = []
            new_object_label_list3 = []

            for object in gt_placeholders_label_new_id[0]:
                if object == 10000:
                    continue
                object_id = new_values_list.index(object)
                new_object_label_list0.append(new_key_list[object_id])

            for object in gt_placeholders_label_new_id[1]:
                if object == 10000:
                    continue
                object_id = new_values_list.index(object)
                new_object_label_list1.append(new_key_list[object_id])

            for object in gt_placeholders_label_new_id[2]:
                if object == 10000:
                    continue
                object_id = new_values_list.index(object)
                new_object_label_list2.append(new_key_list[object_id])

            for object in gt_placeholders_label_new_id[3]:
                if object == 10000:
                    continue
                object_id = new_values_list.index(object)
                new_object_label_list3.append(new_key_list[object_id])

            obj_len_0 = len(new_object_label_list0)
            obj_len_1 = len(new_object_label_list1)
            obj_len_2 = len(new_object_label_list2)
            obj_len_3 = len(new_object_label_list3)

            something = "something"
            something_num = video0_label.count(something)
            Something = "Something"
            Something_num = video0_label.count(Something)
            verb_len_0 = something_num + Something_num

            something = "something"
            something_num = video0_label.count(something)
            Something = "Something"
            Something_num = video0_label.count(Something)
            verb_len_1 = something_num + Something_num

            something = "something"
            something_num = video2_label.count(something)
            Something = "Something"
            Something_num = video2_label.count(Something)
            verb_len_2 = something_num + Something_num

            something = "something"
            something_num = video3_label.count(something)
            Something = "Something"
            Something_num = video3_label.count(Something)
            verb_len_3 = something_num + Something_num

            obj_verb_sentence1 = self.combination(verb_len_1, obj_len_0, video1_label, new_object_label_list0)
            obj_verb_sentence2 = self.combination(verb_len_0, obj_len_1, video0_label, new_object_label_list1)

            obj_verb_sentence3 = self.combination(verb_len_2, obj_len_3, video2_label, new_object_label_list3)
            obj_verb_sentence4 = self.combination(verb_len_3, obj_len_2, video3_label, new_object_label_list2)

            obj_list = [obj_verb_sentence1, obj_verb_sentence2, obj_verb_sentence3, obj_verb_sentence4]

            with torch.no_grad():
                common_pred = process_context(tokenizer, [obj_verb_sentence1, obj_verb_sentence2, obj_verb_sentence3,
                                                          obj_verb_sentence4], common_model, graph)
            is_common = common_pred[0].detach().cpu()[0].tolist()
            is_common_final = []

            for obj in is_common:
                if obj < -1:
                    is_common_final.append(1)
                else:
                    is_common_final.append(0)

            for j in range(len(is_common_final)):
                if is_common_final[j] == 1:
                    self.count += 1

            org_feas = self.backbone(global_img_input)  # [4, 1024, 8, 7, 7]

            b, _, T, H, W = org_feas.size()

            org_feas = org_feas.permute(0, 2, 1, 3, 4).contiguous()

            # swin base 1024
            # org_feas = org_feas.view(b * T, 1024, H, W) # [32, 1024, 7, 7]
            org_feas = org_feas.view(b * T, 768, H, W)
            # print('org_feas:{}'.format(org_feas.shape))

            conv_fea_maps = self.conv(org_feas)  #
            # print('conv_fea_maps.shape', conv_fea_maps.shape)

            box_tensors = box_input.view(b * T, self.nr_boxes, 4)  #
            # print('box_tensors.shape', box_tensors.shape)

            boxes_list = box_to_normalized(box_tensors, crop_size=[224, 224])  # len:16
            # print('boxes_list.shape', len(boxes_list))

            img_size = global_img_input.size()[-2:]

            # (b*T*nr_boxes, C), C=3*3*d
            region_vis_feas = build_region_feas(conv_fea_maps, boxes_list, self.crop_size, img_size)  # [128, 4608]
            # print('region_vis_feas:{}'.format(region_vis_feas.shape))

            region_vis_feas = self.region_vis_embed(region_vis_feas)  # [128, 512]
            # print('2region_vis_feas:{}'.format(region_vis_feas.shape))

            region_vis_feas = region_vis_feas.view(b, T, self.nr_boxes,
                                                   region_vis_feas.size(-1))  # [4, 8, 4, 512]
            # print('3region_vis_feas:{}'.format(region_vis_feas.shape))

            region_vis_feas = region_vis_feas.permute(0, 3, 2, 1).contiguous()  # [4, 512, 4, 8]
            # print('4region_vis_feas:{}'.format(region_vis_feas.shape))

            box_categories = box_categories.cpu().numpy()

            obj1_tensors = torch.zeros((global_img_input.shape[0], 512, 1, 8), dtype=torch.float32)  # (cx, cy, w, h)
            obj2_tensors = torch.zeros((global_img_input.shape[0], 512, 1, 8), dtype=torch.float32)  # (cx, cy, w, h)
            hand_tensors = torch.zeros((global_img_input.shape[0], 512, 1, 8), dtype=torch.float32)  # [4, 512, 1, 8]

            for i in range(global_img_input.shape[0]):
                i_video_cate = box_categories[i]
                for j in range(4):
                    col_indicator = int(np.max(i_video_cate[j]))  # 2,1
                    tmp = region_vis_feas[i, :, j, :]  # [512, 8]
                    tmp = torch.unsqueeze(tmp, 1)  # [512, 1, 8]
                    if col_indicator == 2:
                        obj1_tensors[i] = tmp
                    if col_indicator == 3:
                        obj2_tensors[i] = tmp
                    if col_indicator == 1:
                        hand_tensors[i] = tmp

            global_features = self.avgpool(region_vis_feas).squeeze()  # 全局[4,512]

            concat_obj_0 = torch.cat([obj1_tensors[0], obj2_tensors[0]], 2)  # [512,1,24]
            concat_obj_0 = torch.unsqueeze(concat_obj_0, 0)
            concat_obj_0 = self.avgpool(concat_obj_0).squeeze()
            verb_feature_1 = hand_tensors[0 + 1]  # [512]
            verb_feature_1 = self.avgpool(verb_feature_1).squeeze()
            obj0_ver1 = self.concat_activate(torch.cat([concat_obj_0.cuda(), verb_feature_1.cuda()]))  # [1024]

            concat_obj_1 = torch.cat([obj1_tensors[1], obj2_tensors[1]], 2)
            concat_obj_1 = torch.unsqueeze(concat_obj_1, 0)
            concat_obj_1 = self.avgpool(concat_obj_1).squeeze()
            verb_feature_0 = hand_tensors[0]
            verb_feature_0 = self.avgpool(verb_feature_0).squeeze()
            obj1_ver0 = self.concat_activate(torch.cat([concat_obj_1.cuda(), verb_feature_0.cuda()]))

            concat_obj_2 = torch.cat([obj1_tensors[2], obj2_tensors[2]], 2)  # [512,1,24]
            concat_obj_2 = torch.unsqueeze(concat_obj_2, 0)
            concat_obj_2 = self.avgpool(concat_obj_2).squeeze()
            verb_feature_3 = hand_tensors[2 + 1]  # [512]
            verb_feature_3 = self.avgpool(verb_feature_3).squeeze()
            obj2_ver3 = self.concat_activate(torch.cat([concat_obj_2.cuda(), verb_feature_3.cuda()]))  # [1024]

            concat_obj_3 = torch.cat([obj1_tensors[3], obj2_tensors[3]], 2)
            concat_obj_3 = torch.unsqueeze(concat_obj_3, 0)
            concat_obj_3 = self.avgpool(concat_obj_3).squeeze()
            verb_feature_2 = hand_tensors[2]
            verb_feature_2 = self.avgpool(verb_feature_2).squeeze()
            obj3_ver2 = self.concat_activate(torch.cat([concat_obj_3.cuda(), verb_feature_2.cuda()]))

            com_feature_list = [obj0_ver1, obj1_ver0, obj2_ver3, obj3_ver2]

            is_common_true_feature = []
            is_common_false_feature = []
            true_label = []
            zeros = torch.zeros([1, 1024]).cuda()

            for i in range(len(is_common)):
                if is_common[i] < -1:
                    is_common_true_feature.append(torch.unsqueeze(com_feature_list[i], 0))  # [1,1024]
                    true_label.append(video_label[i])
                else:
                    is_common_false_feature.append(torch.unsqueeze(com_feature_list[i], 0))

            if len(is_common_false_feature) != 4:
                false_len = len(is_common_false_feature)
                for i in range(4 - false_len):
                    is_common_false_feature.append(zeros)

            if len(is_common_true_feature) != 4:
                true_len = len(is_common_true_feature)
                for i in range(4 - true_len):
                    is_common_true_feature.append(zeros)
                    true_label.append(1000)

            is_common_true_feature = torch.cat(is_common_true_feature)
            is_common_false_feature = torch.cat(is_common_false_feature)

            true_label = torch.tensor(true_label).cuda()

            obj0_ver1 = torch.unsqueeze(obj0_ver1, 0)
            obj1_ver0 = torch.unsqueeze(obj1_ver0, 0)
            obj2_ver3 = torch.unsqueeze(obj2_ver3, 0)
            obj3_ver2 = torch.unsqueeze(obj3_ver2, 0)

            is_common_true_feature = torch.unsqueeze(is_common_true_feature, 0)
            is_common_true_feature = self.avgpool1d(is_common_true_feature)
            is_common_true_feature = is_common_true_feature.squeeze(0)
            is_common_true_feature = self.dropout(is_common_true_feature)

            is_common_false_feature = torch.unsqueeze(is_common_false_feature, 0)
            is_common_false_feature = self.avgpool1d(is_common_false_feature)
            is_common_false_feature = is_common_false_feature.squeeze(0)
            is_common_false_feature = self.dropout(is_common_false_feature)

            com_features = torch.cat([obj0_ver1, obj1_ver0, obj2_ver3, obj3_ver2], dim=0)  # [4, 1024]
            com_features = torch.unsqueeze(com_features, 0)
            com_features = self.avgpool1d(com_features)
            com_features = com_features.squeeze(0)
            com_features = self.dropout(com_features)

            com_features = self.dropout(com_features)
            cls_output_com = self.com_fc(com_features)

            global_features = self.dropout(global_features)
            cls_output_glo = self.fc(global_features)

            return cls_output_glo, global_features, cls_output_com, com_features, is_common_true_feature, is_common_false_feature, true_label
