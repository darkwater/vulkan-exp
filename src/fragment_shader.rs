# [ allow ( unused_imports ) ] use std :: sync :: Arc ; # [
allow ( unused_imports ) ] use std :: vec :: IntoIter as VecIntoIter ; # [
allow ( unused_imports ) ] use vulkano :: device :: Device ; # [
allow ( unused_imports ) ] use vulkano :: descriptor :: descriptor ::
DescriptorDesc ; # [ allow ( unused_imports ) ] use vulkano :: descriptor ::
descriptor :: DescriptorDescTy ; # [ allow ( unused_imports ) ] use vulkano ::
descriptor :: descriptor :: DescriptorBufferDesc ; # [
allow ( unused_imports ) ] use vulkano :: descriptor :: descriptor ::
DescriptorImageDesc ; # [ allow ( unused_imports ) ] use vulkano :: descriptor
:: descriptor :: DescriptorImageDescDimensions ; # [ allow ( unused_imports )
] use vulkano :: descriptor :: descriptor :: DescriptorImageDescArray ; # [
allow ( unused_imports ) ] use vulkano :: descriptor :: descriptor ::
ShaderStages ; # [ allow ( unused_imports ) ] use vulkano :: descriptor ::
descriptor_set :: DescriptorSet ; # [ allow ( unused_imports ) ] use vulkano
:: descriptor :: descriptor_set :: UnsafeDescriptorSet ; # [
allow ( unused_imports ) ] use vulkano :: descriptor :: descriptor_set ::
UnsafeDescriptorSetLayout ; # [ allow ( unused_imports ) ] use vulkano ::
descriptor :: pipeline_layout :: PipelineLayout ; # [ allow ( unused_imports )
] use vulkano :: descriptor :: pipeline_layout :: PipelineLayoutDesc ; # [
allow ( unused_imports ) ] use vulkano :: descriptor :: pipeline_layout ::
PipelineLayoutDescPcRange ; # [ allow ( unused_imports ) ] use vulkano ::
pipeline :: shader :: SpecializationConstants as SpecConstsTrait ; # [
allow ( unused_imports ) ] use vulkano :: pipeline :: shader ::
SpecializationMapEntry ; pub struct Shader {
shader : :: std :: sync :: Arc < :: vulkano :: pipeline :: shader ::
ShaderModule > , } impl Shader {
# [ doc = r" Loads the shader in Vulkan as a `ShaderModule`." ] # [ inline ] #
[ allow ( unsafe_code ) ] pub fn load (
device : :: std :: sync :: Arc < :: vulkano :: device :: Device > ) -> Result
< Shader , :: vulkano :: OomError > {
let words = [
119734787u32 , 65536u32 , 851975u32 , 19u32 , 0u32 , 131089u32 , 1u32 ,
393227u32 , 1u32 , 1280527431u32 , 1685353262u32 , 808793134u32 , 0u32 ,
196622u32 , 0u32 , 1u32 , 458767u32 , 4u32 , 4u32 , 1852399981u32 , 0u32 ,
9u32 , 12u32 , 196624u32 , 4u32 , 7u32 , 196611u32 , 2u32 , 450u32 , 589828u32
, 1096764487u32 , 1935622738u32 , 1918988389u32 , 1600484449u32 ,
1684105331u32 , 1868526181u32 , 1667590754u32 , 29556u32 , 655364u32 ,
1197427783u32 , 1279741775u32 , 1885560645u32 , 1953718128u32 , 1600482425u32
, 1701734764u32 , 1919509599u32 , 1769235301u32 , 25974u32 , 524292u32 ,
1197427783u32 , 1279741775u32 , 1852399429u32 , 1685417059u32 , 1768185701u32
, 1952671090u32 , 6649449u32 , 262149u32 , 4u32 , 1852399981u32 , 0u32 ,
327685u32 , 9u32 , 1131705711u32 , 1919904879u32 , 0u32 , 327685u32 , 12u32 ,
1734439526u32 , 1869377347u32 , 114u32 , 262215u32 , 9u32 , 30u32 , 0u32 ,
262215u32 , 12u32 , 30u32 , 0u32 , 131091u32 , 2u32 , 196641u32 , 3u32 , 2u32
, 196630u32 , 6u32 , 32u32 , 262167u32 , 7u32 , 6u32 , 4u32 , 262176u32 , 8u32
, 3u32 , 7u32 , 262203u32 , 8u32 , 9u32 , 3u32 , 262167u32 , 10u32 , 6u32 ,
3u32 , 262176u32 , 11u32 , 1u32 , 10u32 , 262203u32 , 11u32 , 12u32 , 1u32 ,
262187u32 , 6u32 , 14u32 , 1065353216u32 , 327734u32 , 2u32 , 4u32 , 0u32 ,
3u32 , 131320u32 , 5u32 , 262205u32 , 10u32 , 13u32 , 12u32 , 327761u32 , 6u32
, 15u32 , 13u32 , 0u32 , 327761u32 , 6u32 , 16u32 , 13u32 , 1u32 , 327761u32 ,
6u32 , 17u32 , 13u32 , 2u32 , 458832u32 , 7u32 , 18u32 , 15u32 , 16u32 , 17u32
, 14u32 , 196670u32 , 9u32 , 18u32 , 65789u32 , 65592u32 ] ; unsafe {
Ok (
Shader {
shader : r#try ! (
:: vulkano :: pipeline :: shader :: ShaderModule :: from_words (
device , & words ) ) } ) } } # [
doc = r" Returns the module that was created." ] # [ allow ( dead_code ) ] # [
inline ] pub fn module ( & self ) -> & :: std :: sync :: Arc < :: vulkano ::
pipeline :: shader :: ShaderModule > { & self . shader } # [
doc =
r" Returns a logical struct describing the entry point named `{ep_name}`." ] #
[ inline ] # [ allow ( unsafe_code ) ] pub fn main_entry_point ( & self ) ->
:: vulkano :: pipeline :: shader :: GraphicsEntryPoint < (  ) , MainInput ,
MainOutput , Layout > {
unsafe {
# [ allow ( dead_code ) ] static NAME : [ u8 ; 5usize ] = [
109u8 , 97u8 , 105u8 , 110u8 , 0 ] ; self . shader . graphics_entry_point (
:: std :: ffi :: CStr :: from_ptr ( NAME . as_ptr (  ) as * const _ ) ,
MainInput , MainOutput , Layout (
ShaderStages { fragment : true , .. ShaderStages :: none (  ) } ) , :: vulkano
:: pipeline :: shader :: GraphicsShaderType :: Fragment ) } } } # [
derive ( Debug , Copy , Clone , PartialEq , Eq , Hash ) ] pub struct MainInput
; # [ allow ( unsafe_code ) ] unsafe impl :: vulkano :: pipeline :: shader ::
ShaderInterfaceDef for MainInput {
type Iter = MainInputIter ; fn elements ( & self ) -> MainInputIter {
MainInputIter { num : 0 } } } # [ derive ( Debug , Copy , Clone ) ] pub struct
MainInputIter { num : u16 } impl Iterator for MainInputIter {
type Item = :: vulkano :: pipeline :: shader :: ShaderInterfaceDefEntry ; # [
inline ] fn next ( & mut self ) -> Option < Self :: Item > {
if self . num == 0u16 {
self . num += 1 ; return Some (
:: vulkano :: pipeline :: shader :: ShaderInterfaceDefEntry {
location : 0u32 .. 1u32 , format : :: vulkano :: format :: Format ::
R32G32B32Sfloat , name : Some (
:: std :: borrow :: Cow :: Borrowed ( "fragColor" ) ) } ) ; } None } # [
inline ] fn size_hint ( & self ) -> ( usize , Option < usize > ) {
let len = 1usize - self . num as usize ; ( len , Some ( len ) ) } } impl
ExactSizeIterator for MainInputIter {  } # [
derive ( Debug , Copy , Clone , PartialEq , Eq , Hash ) ] pub struct
MainOutput ; # [ allow ( unsafe_code ) ] unsafe impl :: vulkano :: pipeline ::
shader :: ShaderInterfaceDef for MainOutput {
type Iter = MainOutputIter ; fn elements ( & self ) -> MainOutputIter {
MainOutputIter { num : 0 } } } # [ derive ( Debug , Copy , Clone ) ] pub
struct MainOutputIter { num : u16 } impl Iterator for MainOutputIter {
type Item = :: vulkano :: pipeline :: shader :: ShaderInterfaceDefEntry ; # [
inline ] fn next ( & mut self ) -> Option < Self :: Item > {
if self . num == 0u16 {
self . num += 1 ; return Some (
:: vulkano :: pipeline :: shader :: ShaderInterfaceDefEntry {
location : 0u32 .. 1u32 , format : :: vulkano :: format :: Format ::
R32G32B32A32Sfloat , name : Some (
:: std :: borrow :: Cow :: Borrowed ( "outColor" ) ) } ) ; } None } # [ inline
] fn size_hint ( & self ) -> ( usize , Option < usize > ) {
let len = 1usize - self . num as usize ; ( len , Some ( len ) ) } } impl
ExactSizeIterator for MainOutputIter {  } pub mod ty {  } # [
derive ( Debug , Clone ) ] pub struct Layout ( pub ShaderStages ) ; # [
allow ( unsafe_code ) ] unsafe impl PipelineLayoutDesc for Layout {
fn num_sets ( & self ) -> usize { 0usize } fn num_bindings_in_set (
& self , set : usize ) -> Option < usize > { match set { _ => None } } fn
descriptor ( & self , set : usize , binding : usize ) -> Option <
DescriptorDesc > { match ( set , binding ) { _ => None } } fn
num_push_constants_ranges ( & self ) -> usize { 0usize } fn
push_constants_range ( & self , num : usize ) -> Option <
PipelineLayoutDescPcRange > {
if num != 0 || 0usize == 0 { None } else {
Some (
PipelineLayoutDescPcRange {
offset : 0 , size : 0usize , stages : ShaderStages :: all (  ) , } ) } } } # [
derive ( Debug , Copy , Clone ) ] # [ allow ( non_snake_case ) ] # [
repr ( C ) ] pub struct SpecializationConstants {  } impl Default for
SpecializationConstants {
fn default (  ) -> SpecializationConstants { SpecializationConstants {  } } }
unsafe impl SpecConstsTrait for SpecializationConstants {
fn descriptors (  ) -> & 'static [ SpecializationMapEntry ] {
static DESCRIPTORS : [ SpecializationMapEntry ; 0usize ] = [  ] ; &
DESCRIPTORS } }
