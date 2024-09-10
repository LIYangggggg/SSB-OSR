## Running with Higher Resolutions

To run the model with a higher resolution (e.g., 480p), you need to modify the `VisionTransformer` class and comment out the warning code in the `timm` library.


### Step A to Modify:

1. Open the `site-packages/timm/models/vision_transformer.py` file in your environment.
2. Locate the `_pos_embed` function within the `VisionTransformer` class.
3. Find the following code block:

    ```python
    if self.no_embed_class:
        x = x + pos_embed
    ```

4. Insert the following code :

    ```python
    if self.no_embed_class:
        if x.shape[1] != self.pos_embed.shape[1]:
            pass
            old_pos_num = int(math.sqrt(self.pos_embed.shape[1]))
            pos_num =  int(math.sqrt(x.shape[1]))

            self.pos_embed_inter_shape = [pos_num, pos_num]
            self.pos_embed_data = self.pos_embed.data
            self.pos_embed_data = self.pos_embed_data.permute([0, 2, 1])
            self.pos_embed_data = torch.reshape(self.pos_embed_data, (1, 768, old_pos_num, old_pos_num))
            self.pos_embed_data = nn.functional.interpolate(self.pos_embed_data, size=self.pos_embed_inter_shape, mode="bicubic")
            self.pos_embed_data = torch.reshape(self.pos_embed_data, [1, 768, pos_num*pos_num])
            self.pos_embed_data = self.pos_embed_data.permute([0, 2, 1])
            self.pos_embed.data = self.pos_embed_data
        x = x + pos_embed
    ```

### Step B to Ignore:

1. Open the `site-packages/timm/layers/patch_embed.py` file in your environment.
2. Locate the `forward` function within the `PatchEmbed` class.
3. Find the following code block:

    ```python
    if self.img_size is not None:
        if self.strict_img_size:
            _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
            _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
        elif not self.dynamic_img_pad:
            _assert(
                H % self.patch_size[0] == 0,
                f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
            )
            _assert(
                W % self.patch_size[1] == 0,
                f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
            )
    ```

4. Comment out the entire block as shown below:

    ```python
    # if self.img_size is not None:
    #     if self.strict_img_size:
    #         _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
    #         _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
    #     elif not self.dynamic_img_pad:
    #         _assert(
    #             H % self.patch_size[0] == 0,
    #             f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
    #         )
    #         _assert(
    #             W % self.patch_size[1] == 0,
    #             f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
    #         )
    ```

