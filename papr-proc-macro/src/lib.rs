use proc_macro::TokenStream;
use quote::quote;
use syn::{
    braced,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    token::Comma,
    Ident, ItemStruct, Token,
};

struct NodeConstructorParser {
    struc: ItemStruct,
    audio_inputs: Punctuated<Ident, Comma>,
    audio_outputs: Punctuated<Ident, Comma>,
    control_inputs: Punctuated<Ident, Comma>,
    control_outputs: Punctuated<Ident, Comma>,
}

impl Parse for NodeConstructorParser {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let struc: ItemStruct = input.parse()?;

        input.parse::<Token![@]>()?;
        input.parse::<Token![in]>()?;
        let audio_inputs;
        braced!(audio_inputs in input);
        let audio_inputs = audio_inputs.parse_terminated(Ident::parse, Token![,])?;

        input.parse::<Token![@]>()?;
        match input.parse::<Ident>()?.to_string().as_str() {
            "out" => {}
            _ => return Err(input.error("expected `out`")),
        };
        let audio_outputs;
        braced!(audio_outputs in input);
        let audio_outputs = audio_outputs.parse_terminated(Ident::parse, Token![,])?;

        input.parse::<Token![#]>()?;
        input.parse::<Token![in]>()?;
        let control_inputs;
        braced!(control_inputs in input);
        let control_inputs = control_inputs.parse_terminated(Ident::parse, Token![,])?;

        input.parse::<Token![#]>()?;
        match input.parse::<Ident>()?.to_string().as_str() {
            "out" => {}
            _ => return Err(input.error("expected `out`")),
        };
        let control_outputs;
        braced!(control_outputs in input);
        let control_outputs = control_outputs.parse_terminated(Ident::parse, Token![,])?;

        Ok(Self {
            struc,
            audio_inputs,
            audio_outputs,
            control_inputs,
            control_outputs,
        })
    }
}

#[proc_macro]
pub fn node_constructor(tokens: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(tokens as NodeConstructorParser);

    let NodeConstructorParser {
        struc,
        mut audio_inputs,
        audio_outputs,
        mut control_inputs,
        control_outputs,
    } = parsed;

    let struc_name = &struc.ident;

    let control_inputs_args = control_inputs
        .iter()
        .map(|inp| quote! { #inp: crate::Scalar })
        .collect::<Punctuated<_, Comma>>();

    let fields_args = &struc
        .fields
        .iter()
        .map(|f| {
            let ident = f.ident.as_ref().unwrap();
            let ty = &f.ty;
            quote! { #ident: #ty }
        })
        .collect::<Punctuated<_, Comma>>();

    let fields = &struc
        .fields
        .iter()
        .map(|f| {
            let ident = f.ident.as_ref().unwrap();
            quote! { #ident }
        })
        .collect::<Punctuated<_, Comma>>();

    audio_inputs.push(syn::parse2::<Ident>(quote! { t }).unwrap());
    control_inputs.push(syn::parse2::<Ident>(quote! { t }).unwrap());

    let a_outs = audio_outputs
        .iter()
        .map(|out| {
            quote! {
                crate::graph::Output { name: stringify!(#out).to_owned() }
            }
        })
        .collect::<Punctuated<_, Comma>>();

    let c_outs = control_outputs
        .iter()
        .map(|out| {
            quote! {
                crate::graph::Output { name: stringify!(#out).to_owned() }
            }
        })
        .collect::<Punctuated<_, Comma>>();

    let a_ins = audio_inputs
        .iter()
        .map(|inp| {
            quote! {
                crate::graph::Input::new(stringify!(#inp), Some(crate::dsp::Signal::new(0.0)))
            }
        })
        .collect::<Punctuated<_, Comma>>();

    let c_ins = control_inputs
        .iter()
        .map(|inp| {
            quote! {
                crate::graph::Input::new(stringify!(#inp), Some(crate::dsp::Signal::new(0.0)))
            }
        })
        .collect::<Punctuated<_, Comma>>();

    let args = if fields_args.is_empty() {
        quote! { #control_inputs_args }
    } else {
        quote! { #fields_args, #control_inputs_args }
    };

    let audio_input_names_list = audio_inputs
        .iter()
        .map(|inp| {
            quote! { stringify!(#inp) }
        })
        .collect::<Punctuated<_, Comma>>();

    let control_input_names_list = control_inputs
        .iter()
        .map(|inp| {
            quote! { stringify!(#inp) }
        })
        .collect::<Punctuated<_, Comma>>();

    let audio_output_names_list = audio_outputs
        .iter()
        .map(|inp| {
            quote! { stringify!(#inp) }
        })
        .collect::<Punctuated<_, Comma>>();

    let control_output_names_list = control_outputs
        .iter()
        .map(|inp| {
            quote! { stringify!(#inp) }
        })
        .collect::<Punctuated<_, Comma>>();

    let audio_input_mappings = audio_inputs
        .iter()
        .enumerate()
        .map(|(i, inp)| {
            quote! { stringify!(#inp) => {Some(#i)} }
        })
        .collect::<Punctuated<_, Comma>>();

    let control_input_mappings = control_inputs
        .iter()
        .enumerate()
        .map(|(i, inp)| {
            quote! { stringify!(#inp) => {Some(#i)} }
        })
        .collect::<Punctuated<_, Comma>>();

    let audio_output_mappings = audio_outputs
        .iter()
        .enumerate()
        .map(|(i, inp)| {
            quote! { stringify!(#inp) => {Some(#i)} }
        })
        .collect::<Punctuated<_, Comma>>();

    let control_output_mappings = control_outputs
        .iter()
        .enumerate()
        .map(|(i, inp)| {
            quote! { stringify!(#inp) => {Some(#i)} }
        })
        .collect::<Punctuated<_, Comma>>();

    quote! {
        #[derive(Clone)]
        #struc

        #[allow(unused_variables)]
        impl #struc_name {
            pub const AUDIO_INPUTS: &[&'static str] = &[#audio_input_names_list];
            pub const AUDIO_OUTPUTS: &[&'static str] = &[#audio_output_names_list];
            pub const CONTROL_INPUTS: &[&'static str] = &[#control_input_names_list];
            pub const CONTROL_OUTPUTS: &[&'static str] = &[#control_output_names_list];

            pub fn audio_input_idx(name: &str) -> Option<usize> {
                match name {
                    #audio_input_mappings
                    _ => None,
                }
            }

            pub fn audio_output_idx(name: &str) -> Option<usize> {
                match name {
                    #audio_output_mappings
                    _ => None,
                }
            }

            pub fn control_input_idx(name: &str) -> Option<usize> {
                match name {
                    #control_input_mappings
                    _ => None,
                }
            }

            pub fn control_output_idx(name: &str) -> Option<usize> {
                match name {
                    #control_output_mappings
                    _ => None,
                }
            }

            pub fn create_nodes(name: &str, audio_buffer_len: usize, #args) -> (std::sync::Arc<crate::graph::Node<crate::dsp::AudioRate>>, std::sync::Arc<crate::graph::Node<crate::dsp::ControlRate>>) {
                let this = Self { #fields };
                let cn = std::sync::Arc::new(crate::graph::Node::new(
                    crate::graph::NodeName::new(name),
                    1,
                    vec![#c_ins],
                    vec![#c_outs],
                    crate::graph::ProcessorType::Boxed(Box::new(this.clone())),
                    None,
                ));
                let an = std::sync::Arc::new(crate::graph::Node::new(
                    crate::graph::NodeName::new(name),
                    audio_buffer_len,
                    vec![#a_ins],
                    vec![#a_outs],
                    crate::graph::ProcessorType::Boxed(Box::new(this)),
                    Some(cn.clone()),
                ));
                (an, cn)
            }
        }
    }.into()
}