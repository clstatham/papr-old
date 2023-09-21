use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{
    parenthesized,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    token::Comma,
    Ident, ItemStruct, Token,
};

struct TypedIdent {
    ident: Ident,
    ty: Option<Ident>,
}

impl Parse for TypedIdent {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let ident = input.parse()?;
        let ty = if input.peek(Token![:]) {
            input.parse::<Token![:]>()?;
            Some(input.parse()?)
        } else {
            None
        };
        Ok(Self { ident, ty })
    }
}

impl ToTokens for TypedIdent {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let ident = &self.ident;
        // let ty = &self.ty;
        // if let Some(ty) = ty {
        // tokens.extend(quote! { #ident: #ty });
        // } else {
        tokens.extend(quote! { #ident });
        // }
    }
}

struct NodeConstructorParser {
    struc: ItemStruct,
    oversample: bool,
    inputs: Punctuated<TypedIdent, Comma>,
    outputs: Punctuated<TypedIdent, Comma>,
    process_block: Option<syn::Block>,
}

impl Parse for NodeConstructorParser {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let oversample = input.parse::<Token![@]>().is_ok();
        let struc: ItemStruct = input.parse()?;

        let inputs;
        parenthesized!(inputs in input);
        let inputs = inputs.parse_terminated(TypedIdent::parse, Token![,])?;

        input.parse::<Token![->]>()?;
        let outputs;
        parenthesized!(outputs in input);
        let outputs = outputs.parse_terminated(TypedIdent::parse, Token![,])?;

        let process_block = if input.peek(Token![~]) {
            input.parse::<Token![~]>()?;
            Some(input.parse()?)
        } else {
            None
        };

        Ok(Self {
            oversample,
            struc,
            inputs,
            outputs,
            process_block,
        })
    }
}

#[proc_macro]
pub fn node(tokens: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(tokens as NodeConstructorParser);

    let NodeConstructorParser {
        oversample: oversampling,
        struc,
        mut inputs,
        outputs,
        process_block,
    } = parsed;

    let struc_name = &struc.ident;

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

    inputs.push(syn::parse2::<TypedIdent>(quote! { t }).unwrap());

    let outs = outputs
        .iter()
        .map(|out| {
            quote! {
                crate::graph::Output { name: stringify!(#out).to_owned() }
            }
        })
        .collect::<Punctuated<_, Comma>>();

    let ins = inputs
        .iter()
        .map(|inp| {
            if let Some(ty) = &inp.ty {
                quote! {
                    crate::graph::Input::new(stringify!(#inp), Some(crate::dsp::Signal::#ty(Default::default())))
                }
            } else {
                quote! {
                    crate::graph::Input::new(stringify!(#inp), Some(crate::dsp::Signal::Scalar(0.0)))
                }
            }
        })
        .collect::<Punctuated<_, Comma>>();

    let args = if fields_args.is_empty() {
        quote! {}
    } else {
        quote! { #fields_args }
    };

    let input_names_list = inputs
        .iter()
        .map(|inp| {
            quote! { stringify!(#inp) }
        })
        .collect::<Punctuated<_, Comma>>();

    let input_types_list = inputs
        .iter()
        .map(|inp| {
            if let Some(ty) = &inp.ty {
                quote! { crate::dsp::SignalType::#ty }
            } else {
                quote! { crate::dsp::SignalType::Scalar }
            }
        })
        .collect::<Punctuated<_, Comma>>();

    let output_types_list = outputs
        .iter()
        .map(|inp| {
            if let Some(ty) = &inp.ty {
                quote! { crate::dsp::SignalType::#ty }
            } else {
                quote! { crate::dsp::SignalType::Scalar }
            }
        })
        .collect::<Punctuated<_, Comma>>();

    let output_names_list = outputs
        .iter()
        .map(|inp| {
            quote! { stringify!(#inp) }
        })
        .collect::<Punctuated<_, Comma>>();

    let input_mappings = inputs
        .iter()
        .enumerate()
        .map(|(i, inp)| {
            quote! { stringify!(#inp) => {Some(#i)} }
        })
        .collect::<Punctuated<_, Comma>>();

    let output_mappings = outputs
        .iter()
        .enumerate()
        .map(|(i, inp)| {
            quote! { stringify!(#inp) => {Some(#i)} }
        })
        .collect::<Punctuated<_, Comma>>();

    let process_block = if let Some(process_block) = process_block {
        let input_bindings = inputs
            .iter()
            .enumerate()
            .map(|(i, inp)| {
                let ident = &inp.ident;
                let ty = &inp.ty;
                if let Some(ty) = ty {
                    let ty = ty.to_string().to_lowercase();
                    let expect = syn::parse_str::<syn::Path>(&format!("expect_{}", ty)).unwrap();
                    quote! { let #ident = &inputs[#i].#expect()?; }
                } else {
                    quote! { let #ident = &inputs[#i].expect_scalar()?; }
                }
            })
            .collect::<proc_macro2::TokenStream>();
        let output_bindings = outputs
            .iter()
            .map(|out| {
                let ident = &out.ident;
                quote! { let mut #ident; }
            })
            .collect::<proc_macro2::TokenStream>();
        let output_assignments = outputs
            .iter()
            .enumerate()
            .map(|(i, out)| {
                let ident = &out.ident;
                let ty = &out.ty;
                if let Some(ty) = ty {
                    quote! { outputs[#i] = crate::dsp::Signal::#ty(#ident); }
                } else {
                    quote! { outputs[#i] = crate::dsp::Signal::Scalar(#ident); }
                }
            })
            .collect::<proc_macro2::TokenStream>();
        let block = quote! {
            #[allow(unused_variables)]
            #[allow(unused_mut)]
            #[allow(unused_assignments)]
            #[allow(unreachable_code)]
            impl crate::dsp::Processor for #struc_name {
                fn process_sample(
                    &mut self,
                    buffer_idx: usize,
                    signal_rate: crate::dsp::SignalRate,
                    inputs: &[crate::dsp::Signal],
                    outputs: &mut [crate::dsp::Signal],
                ) -> miette::Result<()> {
                    #input_bindings
                    #output_bindings
                    #process_block
                    #output_assignments
                    Ok(())
                }
            }
        };
        block
    } else {
        quote! {}
    };

    let node_finalize = if oversampling {
        quote! { crate::dsp::oversampling::Oversample2x::create_node(n) }
    } else {
        quote! { n }
    };

    quote! {
        #[derive(Clone)]
        #struc

        #process_block

        #[allow(unused_variables)]
        impl #struc_name {
            pub const INPUTS: &'static [&'static str] = &[#input_names_list];
            pub const OUTPUTS: &'static [&'static str] = &[#output_names_list];

            pub const INPUT_TYPES: &'static [crate::dsp::SignalType] = &[#input_types_list];
            pub const OUTPUT_TYPES: &'static [crate::dsp::SignalType] = &[#output_types_list];

            pub fn validate_inputs(inputs: &[crate::dsp::Signal]) -> bool {
                inputs.len() == Self::INPUTS.len() && inputs.iter().zip(Self::INPUT_TYPES.iter()).all(|(a, b)| {
                    match (a, b) {
                        (crate::dsp::Signal::Scalar(_), crate::dsp::SignalType::Scalar) => true,
                        (crate::dsp::Signal::Array(_), crate::dsp::SignalType::Array) => true,
                        (crate::dsp::Signal::Symbol(_), crate::dsp::SignalType::Symbol) => true,
                        _ => false,
                    }
                })
            }

            pub fn validate_outputs(outputs: &[crate::dsp::Signal]) -> bool {
                outputs.len() == Self::OUTPUTS.len() && outputs.iter().zip(Self::OUTPUT_TYPES.iter()).all(|(a, b)| {
                    match (a, b) {
                        (crate::dsp::Signal::Scalar(_), crate::dsp::SignalType::Scalar) => true,
                        (crate::dsp::Signal::Array(_), crate::dsp::SignalType::Array) => true,
                        (crate::dsp::Signal::Symbol(_), crate::dsp::SignalType::Symbol) => true,
                        _ => false,
                    }
                })
            }

            pub fn input_idx(name: &str) -> Option<usize> {
                match name {
                    #input_mappings
                    _ => None,
                }
            }

            pub fn output_idx(name: &str) -> Option<usize> {
                match name {
                    #output_mappings
                    _ => None,
                }
            }


            #[allow(clippy::too_many_arguments)]
            pub fn create_node(name: &str, #args) -> std::sync::Arc<crate::graph::Node> {
                let this = Box::new(std::sync::RwLock::new(Self { #fields }));
                let n = crate::graph::Node::new(
                    crate::graph::NodeName::new(name),
                    vec![#ins],
                    vec![#outs],
                    crate::graph::ProcessorType::Builtin(this),
                );
                let n = std::sync::Arc::new(n);
                #node_finalize
            }
        }
    }
    .into()
}
